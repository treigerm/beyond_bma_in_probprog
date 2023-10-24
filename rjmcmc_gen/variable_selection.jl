using Gen
using CSV
using DataFrames
using ProgressMeter
using Random
using Pickle
using Base.Filesystem
using LinearAlgebra

const INCLUSION_PROB = 0.5
const GAMMA_CONCENTRATION = 2.0
const GAMMA_RATE = 1.0
const WEIGHT_LAMBDA = 1.0

struct VecBernoulli <: Distribution{Vector{Bool}} end

"""
    bernoulli(prob_true::Vector{Real})

Samples a vector of `Bool` values which is true with given probability
"""
const vecbernoulli = VecBernoulli()

function Gen.logpdf(::VecBernoulli, x::Vector{Bool}, prob::Vector{T}) where T <: Real
    return sum(log.(prob) .* x .+ log.(1. .- prob) .* (1 .- x))
end

# function logpdf_grad.(::VecBernoulli, x::Vector{Bool}, prob::Vector{Real})
#     prob_grad = x ? 1. / prob : -1. / (1-prob)
#     (nothing, prob_grad)
# end

Gen.random(::VecBernoulli, prob::Vector{T}) where {T <: Real} = rand(size(prob, 1)) .< prob

Gen.is_discrete(::VecBernoulli) = true

(::VecBernoulli)(prob) = random(VecBernoulli(), prob)

Gen.has_output_grad(::VecBernoulli) = false
Gen.has_argument_grads(::VecBernoulli) = (false,)

sigmoid(x) = 1 / (1 + exp(-x))

function chmap_to_dict(chmap, num_features)
    res = Dict()
    for ix in 1:num_features
        res[("feature", ix)] = chmap[(:feature, ix)]
    end
    num_selected = sum(chmap[(:feature, ix)] for ix in 1:num_features)
    num_selected = max(num_selected, 1)
    for ix in 1:num_selected
        res[("weight", ix)] = chmap[(:weight, ix)]
    end
    if has_value(chmap, :noise_var)
        res["noise_var"] = chmap[:noise_var]
    end
    res
end

@gen function variable_selection(X, y)
    num_features = size(X, 2)
    features_included = zeros(Bool, num_features)
    for ix in 1:num_features
        features_included[ix] = {(:feature, ix)} ~ bernoulli(INCLUSION_PROB)
    end

    if any(features_included)
        X_selected = X[:, features_included]
    else
        X_selected = ones(size(X, 1), 1)
    end

    num_selected = size(X_selected, 2)
    noise_var ~ inv_gamma(GAMMA_CONCENTRATION, GAMMA_RATE)
    weights = zeros(num_selected)
    for ix in 1:num_selected
        weights[ix] = {(:weight, ix)} ~ normal(0, sqrt(noise_var / WEIGHT_LAMBDA))
    end

    means = X_selected * weights
    # for ix in 1:size(y, 1)
    #     {(:y, ix)} ~ normal(means[ix], sqrt(noise_var))
    # end
    y ~ mvnormal(means, Diagonal(noise_var * ones(size(y, 1))))
end

@gen function log_reg(X, y)
    num_features = size(X, 2)
    features_included = zeros(Bool, num_features)
    for ix in 1:num_features
        features_included[ix] = {(:feature, ix)} ~ bernoulli(INCLUSION_PROB)
    end

    if any(features_included)
        X_selected = X[:, features_included]
    else
        X_selected = ones(size(X, 1), 1)
    end

    num_selected = size(X_selected, 2)
    weights = zeros(num_selected)
    for ix in 1:num_selected
        weights[ix] = {(:weight, ix)} ~ normal(0, 1)
    end

    probs = sigmoid.(X_selected * weights)
    # for ix in 1:size(y, 1)
    #     {(:y, ix)} ~ bernoulli(probs[ix])
    # end
    y ~ vecbernoulli(probs)
end

function get_weight_ix(tr, flip_feature, num_features)
    num_features_included = sum(tr[(:feature, ix)] for ix in 1:num_features)
    weight_ix = 1
    # Find the index of the flipped feature in the parameter vector.
    for ix in 1:num_features
        if ix == flip_feature
            break
        end
        if tr[(:feature, ix)]
            weight_ix += 1
        end
    end
    return weight_ix
end

@gen function jump_proposal(tr, X, y) 
    num_features = size(X, 2)
    flip_feature ~ uniform_discrete(1, num_features)
    features_is_included = tr[(:feature, flip_feature)]

    num_features_included = sum(tr[(:feature, ix)] for ix in 1:num_features)
    weight_ix = get_weight_ix(tr, flip_feature, num_features)
    if features_is_included
        # Update the weight for all remaining weights. Ignore flipped weight
        for ix in 1:num_features_included
            if ix != weight_ix
                {(:new_weight, ix)} ~ normal(tr[(:weight, ix)], 1.0)
            end
        end
    else
        for ix in 1:num_features_included
            {(:new_weight, ix)} ~ normal(tr[(:weight, ix)], 1.0)
        end
        new_feature_weight ~ normal(0, 1.0)
    end
end

function involution(trace, forward_choices, forward_retval, proposal_args)
    backward_choices = choicemap()

    num_features = size(proposal_args[1], 2)
    num_features_included = sum(trace[(:feature, ix)] for ix in 1:num_features)
    feature_is_included = trace[(:feature, forward_choices[:flip_feature])]
    backward_choices[:flip_feature] = forward_choices[:flip_feature]
    weight_ix = get_weight_ix(trace, forward_choices[:flip_feature], num_features)
    if feature_is_included
        # We need to go backwards from the feature being excluded to the feature 
        # being included. Means we need to insert a new weight at the correct index.
        backward_choices[:new_feature_weight] = trace[(:weight, weight_ix)]

        # Current trace has num_features_included weights. However, since we will
        # be moving from a trace that has num_features_included-1 weights, we need 
        # to adjust the indices in the backward_choices.
        for ix in 1:num_features_included
            if ix < weight_ix
                backward_choices[(:new_weight, ix)] = trace[(:weight, ix)]
            elseif ix == weight_ix
                continue
            elseif ix > weight_ix
                backward_choices[(:new_weight, ix - 1)] = trace[(:weight, ix)]
            end
        end
    else
        # We go backwards from a trace that has the feature included to a trace that 
        # has the feature excluded. Need to only update the weights which are still in
        # the new trace.
        for ix in 1:(num_features_included + 1)
            if ix < weight_ix
                backward_choices[(:new_weight, ix)] = trace[(:weight, ix)]
            elseif ix == weight_ix
                continue
            elseif ix > weight_ix
                backward_choices[(:new_weight, ix)] = trace[(:weight, ix - 1)]
            end
        end
    end

    new_trace_choices = choicemap()
    new_trace_choices[(:feature, forward_choices[:flip_feature])] = !feature_is_included
    if feature_is_included
        # Remove the weight from the current trace.
        for ix in 1:(num_features_included - 1)
            if ix < weight_ix
                new_trace_choices[(:weight, ix)] = trace[(:weight, ix)]
            elseif ix == weight_ix
                continue
            elseif ix > weight_ix
                new_trace_choices[(:weight, ix - 1)] = trace[(:weight, ix)]
            end
        end
    else
        for ix in 1:(num_features_included + 1)
            if ix < weight_ix
                new_trace_choices[(:weight, ix)] = trace[(:weight, ix)]
            elseif ix == weight_ix
                new_trace_choices[(:weight, ix)] = forward_choices[:new_feature_weight]
            elseif ix > weight_ix
                new_trace_choices[(:weight, ix)] = trace[(:weight, ix - 1)]
            end
        end
    end

    # Obtain an updated trace matching the choicemap, and a weight
    new_trace, weight, = update(trace, new_trace_choices)
    (new_trace, backward_choices, weight)
end

function custom_update_inv(tr, X, y)
    tr, accepted = mh(tr, jump_proposal, (X, y), involution)
    if has_value(get_choices(tr), :noise_var)
        # Log-reg does not have a noise_var.
        tr, = mh(tr, Gen.select(:noise_var))
    end
    tr
end

function do_inference(model, X, y, num_iters)
    observations = Gen.choicemap()
    # for (i, y_val) in enumerate(y)
    #     observations[(:y, i)] = y_val
    # end
    observations[:y] = y

    trace, = generate(model, (X, y), observations)
    traces = []
    num_features = size(X, 2)
    @showprogress 1 "Sampling..." for i in 1:num_iters
        trace = custom_update_inv(trace, X, y)
        push!(
            traces, 
            # trace
            chmap_to_dict(get_choices(trace), num_features)
        )
    end
    traces
end

function main(data_dir, fname, do_log_reg; num_samples=100)
    data_path = joinpath(data_dir, "data.pickle")
    (
        X_train,
        y_train,
        X_val, 
        y_val,
        _, 
        _
    ) = Pickle.Torch.THload(data_path)
    X = vcat(X_train, X_val)
    y = vcat(y_train, y_val)
    model = do_log_reg ? log_reg : variable_selection
    if do_log_reg
        y = convert(Vector{Bool}, y)
    end
    traces = do_inference(model, X, y, num_samples)

    fname = joinpath("results", "variable_selection", "$fname.pickle")
    store(
        fname, 
        Dict(
            "traces" => traces
            # "traces" => map(
            #     x -> chmap_to_dict(get_choices(x), X, y), 
            #     traces
            # )
        )
    )
end

function all_main(dir_name, dataname; num_samples=100, dname=nothing)
    items = readdir(dir_name)
    directories = filter(x -> isdir(joinpath(dir_name, x)), items)
    do_log_reg = dataname in ["diabetes", "stroke"]
    if dname === nothing
        for dname in directories
            main(
                joinpath(dir_name, dname), 
                "$(dataname)_$(dname)",
                do_log_reg; 
                num_samples=num_samples,
            )
        end
    else
        if !(dname in directories)
            println("Directory $dname not found in $dir_name")
            return
        end
        main(
            joinpath(dir_name, dname), 
            "$(dataname)_$(dname)",
            do_log_reg; 
            num_samples=num_samples,
        )
    end
end