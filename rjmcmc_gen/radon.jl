using Gen
using CSV
using DataFrames
using ProgressMeter
using Random
using Pickle
using Base.Filesystem

const DATA_DIR = "radon/"

function chmap_to_dict(chmap)
    result = Dict()
    top_keys = [:alpha_choice, :beta_choice, :sigma]
    for key in top_keys
        result[string(key)] = chmap[key]
    end

    alpha_choice = chmap[:alpha_choice]
    beta_choice = chmap[:beta_choice]

    alpha_submap = get_submap(chmap, (:alpha, alpha_choice))
    beta_submap = get_submap(chmap, (:beta, beta_choice))

    for (key, val) in get_values_shallow(alpha_submap)
        if isa(key, Tuple)
            result[(string(key[1]), key[2])] = val
        else
            result[string(key)] = val
        end
    end
    for (key, val) in get_values_shallow(beta_submap)
        if isa(key, Tuple)
            result[(string(key[1]), key[2])] = val
        else
            result[string(key)] = val
        end
    end
    result
end

function reshape_param(params, county)
    result = Array{Real}(undef, size(county, 1))
    for i in 1:size(county, 1)
        result[i] = params[county[i]]
    end
    result
end

function factorize(df::DataFrame, column_name::Symbol)
    column_data = df[!, column_name]
    unique_values = unique(column_data)
    index_dict = Dict(unique_values[i] => i for i in 1:length(unique_values))
    
    numerical_indices = [index_dict[val] for val in column_data]
    
    return numerical_indices, unique_values
end

function stratified_train_test_split(log_radon, floor_measure, county)
    # Make dictionary of counties and their indices
    unique_counties = Set(county)
    county_dict = Dict{eltype(county), Vector{Int}}()
    for c in unique_counties
        county_dict[c] = findall(x -> x == c, county)
    end

    training_ixs = Vector{Int}[]
    test_ixs = Vector{Int}[]

    for ixs in values(county_dict)
        # Stratified sampling
        n = length(ixs)
        n_train = max(floor(Int, n * 0.8), 1)
        shuffle!(ixs)
        push!(training_ixs, ixs[1:n_train])
        push!(test_ixs, ixs[n_train+1:end])
    end

    training_ixs = vcat(training_ixs...)
    test_ixs = vcat(test_ixs...)

    log_radon_train = log_radon[training_ixs]
    log_radon_test = log_radon[test_ixs]
    floor_measure_train = floor_measure[training_ixs]
    floor_measure_test = floor_measure[test_ixs]
    county_train = county[training_ixs]
    county_test = county[test_ixs]

    return (
        log_radon_train,
        log_radon_test,
        floor_measure_train,
        floor_measure_test,
        county_train,
        county_test,
    )
end

function load_data()
    # Assuming MY_DIR is defined as your directory path
    srrs2 = DataFrame(CSV.File(joinpath(DATA_DIR, "srrs2.dat")))
    # srrs2 = select!(srrs2, Not(:xstate))
    
    srrs_mn = filter(row -> row.state == "MN", srrs2)
    srrs_mn.fips = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
    
    cty = DataFrame(CSV.File(joinpath(DATA_DIR, "cty.dat")))
    cty_mn = filter(row -> row.st == "MN", cty)
    
    cty_mn.fips = 1000 * cty_mn.stfips + cty_mn.ctfips
    
    srrs_mn = innerjoin(srrs_mn, cty_mn[:, [:fips, :Uppm]], on=:fips)
    srrs_mn = unique(srrs_mn, :idnum)
    
    uranium = unique(log.(srrs_mn.Uppm))
    
    srrs_mn.county .= strip.(srrs_mn.county)
    county, mn_counties = factorize(srrs_mn, :county)
    srrs_mn.county_code = county
    
    radon = srrs_mn.activity
    log_radon = log.(radon .+ 0.1)
    floor_measure = srrs_mn.floor

    (log_radon_train, log_radon_test, floor_measure_train, floor_measure_test, county_train, county_test) =
        stratified_train_test_split(log_radon, floor_measure, county)
    
    return (
        log_radon_train,
        log_radon_test,
        floor_measure_train,
        floor_measure_test,
        county_train,
        county_test,
        uranium,
        mn_counties,
    )
end

@gen function alpha_pooled_prior(county, num_counties, uranium)
    alpha ~ normal(0, 10)
    alpha = fill(alpha, length(county))
    alpha
end

@gen function alpha_county_specific_prior(county, num_counties, uranium)
    alpha = Vector{Real}(undef, num_counties)
    for i in 1:num_counties
        alpha[i] = {(:alpha, i)} ~ normal(0, 10)
    end
    alpha = reshape_param(alpha, county)
    alpha
end

@gen function alpha_partially_pooled_prior(county, num_counties, uranium)
    mean_a ~ normal(0, 1)
    std_a ~ exponential(1)
    z_a = Vector{Real}(undef, num_counties)
    for i in 1:num_counties
        z_a[i] = {(:z_a, i)} ~ normal(0, 1)
    end
    alpha = mean_a .+ std_a .* z_a
    alpha = reshape_param(alpha, county)
    alpha
end

@gen function alpha_uranium_context(county, num_counties, uranium)
    # Uranium context
    gamma_0 ~ normal(0, 10)
    gamma_1 ~ normal(0, 10)
    mean_a = gamma_0 .+ gamma_1 .* uranium
    std_a ~ exponential(1)
    z_a = Vector{Real}(undef, num_counties)
    for i in 1:num_counties
        z_a[i] = {(:z_a, i)} ~ normal(0, 1)
    end
    alpha = mean_a .+ std_a .* z_a
    alpha = reshape_param(alpha, county)
    alpha
end

@gen function beta_pooled_prior(county, num_counties)
    betas ~ normal(0, 10)
    betas = fill(betas, length(county))
    betas
end

@gen function beta_county_specific_prior(county, num_counties)
    betas = Vector{Real}(undef, num_counties)
    for i in 1:num_counties
        betas[i] = {(:beta, i)} ~ normal(0, 10)
    end
    betas = reshape_param(betas, county)
    betas
end

@gen function beta_partially_pooled_model(county, num_counties)
    mean_b ~ normal(0, 1)
    std_b ~ exponential(1)
    z_b = Vector{Real}(undef, num_counties)
    for i in 1:num_counties
        z_b[i] = {(:z_b, i)} ~ normal(0, 1)
    end
    betas = mean_b .+ std_b .* z_b
    betas = reshape_param(betas, county)
    betas
end


@gen function radon_model(log_radon, floor_ind, county, num_counties, uranium)
    alpha_choice ~ uniform_discrete(1, 4)
    if alpha_choice == 1
        alpha = {(:alpha, alpha_choice)} ~ alpha_pooled_prior(county, num_counties, uranium)
    elseif alpha_choice == 2
        alpha = {(:alpha, alpha_choice)} ~ alpha_county_specific_prior(county, num_counties, uranium)
    elseif alpha_choice == 3
        alpha = {(:alpha, alpha_choice)} ~ alpha_partially_pooled_prior(county, num_counties, uranium)
    elseif alpha_choice == 4
        alpha = {(:alpha, alpha_choice)} ~ alpha_uranium_context(county, num_counties, uranium)
    end

    beta_choice ~ uniform_discrete(1, 3)
    if beta_choice == 1
        betas = {(:beta, beta_choice)} ~ beta_pooled_prior(county, num_counties)
    elseif beta_choice == 2
        betas = {(:beta, beta_choice)} ~ beta_county_specific_prior(county, num_counties)
    elseif beta_choice == 3
        betas = {(:beta, beta_choice)} ~ beta_partially_pooled_model(county, num_counties)
    end

    theta = alpha .+ betas .* floor_ind
    sigma ~ exponential(5)
    for i in 1:size(log_radon, 1)
        {(:y, i)} ~ normal(theta[i], sigma)
    end
end

@gen function jump_proposal(tr, log_radon, floor_ind, county, num_counties, uranium)
    alpha_choice ~ uniform_discrete(1, 4)
    ALPHA_PRIORS = [
        alpha_pooled_prior,
        alpha_county_specific_prior,
        alpha_partially_pooled_prior,
        alpha_uranium_context,
    ]
    alpha ~ ALPHA_PRIORS[alpha_choice](county, num_counties, uranium)

    beta_choice ~ uniform_discrete(1, 3)
    BETA_PRIOR = [
        beta_pooled_prior,
        beta_county_specific_prior,
        beta_partially_pooled_model,
    ]
    beta ~ BETA_PRIOR[beta_choice](county, num_counties)
end

function involution(trace, forward_choices, forward_retval, proposal_args)
    # Goal: based on `forward_choices`, return a new trace
    # and the `backward_choices` required to reverse this
    # proposal.
    new_trace_choices = choicemap()
    backward_choices  = choicemap()

    # Copy all the values from trace into backward_choices, except for :sigma.
    backward_choices[:alpha_choice] = trace[:alpha_choice]
    set_submap!(backward_choices, :alpha, get_submap(get_choices(trace), (:alpha, trace[:alpha_choice])))
    backward_choices[:beta_choice] = trace[:beta_choice]
    set_submap!(backward_choices, :beta, get_submap(get_choices(trace), (:beta, trace[:beta_choice])))

    new_trace_choices[:alpha_choice] = forward_choices[:alpha_choice]
    set_submap!(
        new_trace_choices, 
        (:alpha, forward_choices[:alpha_choice]), 
        get_submap(forward_choices, :alpha)
    )
    new_trace_choices[:beta_choice] = forward_choices[:beta_choice]
    set_submap!(
        new_trace_choices, 
        (:beta, forward_choices[:beta_choice]), 
        get_submap(forward_choices, :beta)
    )

    # Obtain an updated trace matching the choicemap, and a weight
    new_trace, weight, = update(trace, new_trace_choices)
    (new_trace, backward_choices, weight)
end

function custom_update_inv(tr, log_radon, floor_ind, county, num_counties, uranium)
    tr, accepted = mh(tr, jump_proposal, (log_radon, floor_ind, county, num_counties, uranium), involution)
    tr, = mh(tr, Gen.select(:sigma))
    tr
end

function do_inference(log_radon, floor_ind, county, num_counties, uranium, num_iters)
    observations = Gen.choicemap()
    for (i, y) in enumerate(log_radon)
        observations[(:y, i)] = y
    end

    trace, = generate(radon_model, (log_radon, floor_ind, county, num_counties, uranium), observations)
    traces = []
    @showprogress 1 "Sampling..." for i in 1:num_iters
        trace = custom_update_inv(trace, log_radon, floor_ind, county, num_counties, uranium)
        push!(traces, trace)
    end
    traces
end


function main(data_dir; num_samples=100, fname="results/radon/radon_traces.pkl")
    data_path = joinpath(data_dir, "data.pickle")
    counties_path = joinpath(data_dir, "mn_counties.pickle")
    (
        log_radon_train, 
        log_radon_test, 
        floor_measure_train, 
        floor_measure_test, 
        county_train, 
        county_test, 
        uranium
    ) = Pickle.Torch.THload(data_path)
    # Convert to 1-based indexing.
    county_train = county_train .+ 1
    county_test = county_test .+ 1
    mn_counties = Pickle.load(open(counties_path))
    num_counties = length(mn_counties)

    # Run inference
    traces = do_inference(log_radon_train, floor_measure_train, county_train, num_counties, uranium, num_samples)
    
    # Convert traces to Dicts
    traces_dict = []
    for tr in traces
        traces_dict = push!(traces_dict, chmap_to_dict(get_choices(tr)))
    end

    store(fname, Dict("traces" => traces_dict))
end

function all_main(dir_name; num_samples=100)
    items = readdir(dir_name)
    directories = filter(x -> isdir(joinpath(dir_name, x)), items)
    for dname in directories
        main(
            joinpath(dir_name, dname); 
            num_samples=num_samples,
            fname="results/radon/radon_traces_$(dname).pkl"
        )
    end
end
