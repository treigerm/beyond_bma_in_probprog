using Gen
using ProgressMeter
using MAT
using Distributions
using Pickle

const FULL_DATA_PATH = "" # TODO: Fill out path to data generated from ../subset_regression.py

@gen function subset_regression(xs)
    k ~ uniform_discrete(1, 15)
    beta = {(:beta, k)} ~ normal(0, sqrt(10))
    sigma ~ gamma(0.1, 10.0)
    mean = beta * xs[:, k]
    for i in 1:size(xs, 1)
        {(:y, i)} ~ normal(mean[i], sigma)
    end
end

@gen function jump_proposal(tr, xs, ys)
    new_k ~ uniform_discrete(1, 15)
    # mean_guess = xs[:, new_k]' * ys / (xs[:, new_k]' * xs[:, new_k])
    # new_beta ~ normal(mean_guess, 1.0)
    new_beta ~ normal(tr[(:beta, tr[:k])], 1.0)
    # new_beta ~ normal(0, sqrt(10))
end

function involution(t, forward_choices, forward_retval, proposal_args)
    # Goal: based on `forward_choices`, return a new trace
    # and the `backward_choices` required to reverse this
    # proposal.
    new_trace_choices = choicemap()
    backward_choices  = choicemap()

    backward_choices[:new_k] = t[:k]
    backward_choices[:new_beta] = t[(:beta, t[:k])]

    new_trace_choices[:k] = forward_choices[:new_k]
    new_trace_choices[(:beta, forward_choices[:new_k])] = forward_choices[:new_beta]

    # Obtain an updated trace matching the choicemap, and a weight
    new_trace, weight, = update(t, get_args(t), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end

function custom_update_inv(tr, xs, ys)
    tr, accepted = mh(tr, jump_proposal, (xs, ys), involution)
    tr, = mh(tr, select(:sigma))
    tr
end

function do_inference(xs, ys, num_iters)
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    trace, = generate(subset_regression, (xs,), observations)
    traces = []
    @showprogress 1 "Sampling..." for i in 1:num_iters
        trace = custom_update_inv(trace, xs, ys)
        push!(traces, trace)
    end
    traces
end

function coeffs(js, x, h)
    return (h .- abs.(js .- x)).^2 .* (abs.(js .- x) .< h)
end


function get_coefficients(;J=15, h=5)
    js = collect(1:J)
    alphas = coeffs(js, 4, h) + coeffs(js, 8, h) + coeffs(js, 12, h)
    gamma = sqrt(4 / sum(alphas.^2))
    betas = gamma * alphas
    return betas
end


function generate_data(;N_train=200, N_val=200, N_test=200, h=5, J=15)
    betas = get_coefficients(J=J, h=h)

    X_train = 5 .+ randn(N_train, J) 
    y_train = X_train * betas .+ randn(N_train)

    X_test = 5 .+ randn(N_test, J)
    y_test = X_test * betas .+ randn(N_test)

    (X_train, y_train, X_test, y_test)
end

function evaluate_log_post_pred(trace, X_test, y_test)
    k = trace[:k]
    beta = trace[(:beta, k)]
    sigma = trace[:sigma]
    # Return (num_test,)
    Distributions.logpdf.(Normal.(X_test[:, k] * beta, sigma), y_test)
end

function lppd(traces, X_test, y_test)
    lppds = zeros(Float64, length(traces), size(X_test, 1))
    for (ix, trace) in enumerate(traces)
        # push!(lppds, evaluate_log_post_pred(trace, X_test, y_test))
        lppds[ix, :] = evaluate_log_post_pred(trace, X_test, y_test)
    end
    data_point_lppd = zeros(Float64, length(traces))
    for i in 1:size(X_test, 1)
        data_point_lppd = logsumexp(lppds[:,i]) - log(length(traces))
    end
    mean(data_point_lppd)
end

function traces_to_samples(traces)
    samples = Dict(
        "k" => [], "beta" => [], "sigma" => []
    )
    for (ix, tr) in enumerate(traces)
        push!(samples["k"], tr[:k])
        push!(samples["beta"], tr[(:beta, tr[:k])])
        push!(samples["sigma"], tr[:sigma])
    end
    samples
end

function main(;num_samples=1_000, num_warmup=0, fname="rjmcmc_results.pickle")
    data = matread(FULL_DATA_PATH)
    lppds = Float64[]
    weights = Array{Float64, 1}[]
    samples = []
    for (iter_ix, data_splits) in data
        X_train = data_splits["X_train"]
        y_train = vec(data_splits["y_train"])
        X_test = data_splits["X_test"]
        y_test = vec(data_splits["y_test"])
        traces = do_inference(X_train, y_train, num_samples);
        traces = traces[num_warmup+1:end]
        slp_count = zeros(Float64, 15)
        for tr in traces
            slp_count[tr[:k]] += 1
        end
        push!(weights, slp_count / sum(slp_count))
        push!(lppds, lppd(traces, X_test, y_test))
        push!(samples, traces_to_samples(traces))
    end
    @show lppds
    @show weights
    store(fname, Dict("lppds" => lppds, "weights" => weights, "samples" => samples))
end