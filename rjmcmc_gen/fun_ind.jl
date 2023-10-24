using Gen
using CSV
using DataFrames
using ProgressMeter
using Random
using Pickle
using Distributions
using Base.Filesystem
using LinearAlgebra

abstract type Expr end
abstract type PrimitiveExpr <: Expr end
abstract type CompositeExpr <: Expr end

struct Identity <: PrimitiveExpr
    a::Float64
end

eval_fn(expr::Identity, x) = x

struct Sine <: CompositeExpr 
    subexpr::Expr
    a::Float64
end

eval_fn(expr::Sine, x) = sin(expr.a * eval_fn(expr.subexpr, x))

# struct Quadratic <: PrimitiveExpr
#     a::Float64
# end

# eval_fn(expr::Quadratic, x) = x^2

struct Add <: CompositeExpr
    left::Expr
    right::Expr
    a::Float64
    b::Float64
end

eval_fn(expr::Add, x) = expr.a * eval_fn(expr.left, x) + expr.b * eval_fn(expr.right, x)

expr_types = [Identity, Sine, Add]
@dist choose_expr_type() = expr_types[categorical([0.4, 0.4, 0.2])]

@gen function sample_expression()
    expr_type ~ choose_expr_type()
    if expr_type == Add
        return expr_type(
            {:left} ~ sample_expression(), 
            {:right} ~ sample_expression(), 
            {:a} ~ normal(0, 10),
            {:b} ~ normal(0, 10)
        )
    elseif expr_type == Sine
        return expr_type(
            {:subexpr} ~ sample_expression(),
            {:a} ~ normal(0, 10)
        )
    elseif expr_type == Identity
        return expr_type({:a} ~ normal(0, 10))
    end
end

@gen function model(xs:Vector{Float64}) 
    sampled_expr = {:tree} ~ sample_expression()
    means = Vector{Float64}(undef, length(xs))
    for ix in 1:length(xs)
        means[ix] = eval_fn(sampled_expr, xs[ix])
    end
    std ~ gamma(1.0, 1.0)
    # for (i, x) in enumerate(xs)
    #     {(:y, i)} ~ normal(means[i], std)
    # end
    y ~ mvnormal(means, Diagonal(std * ones(size(xs, 1))))
    return sampled_expr
end

@gen function random_node_path(n::Expr)
    if ({:stop} ~ bernoulli(isa(n, PrimitiveExpr) ? 1.0 : 0.5))
        return :tree
    else
        if isa(n, Add)
            (next_node, direction) = ({:left} ~ bernoulli(0.5)) ? (n.left, :left) : (n.right, :right)
        elseif isa(n, Sine)
            (next_node, direction) = (n.subexpr, :subexpr)
        end
        rest_of_path ~ random_node_path(next_node)
        
        if isa(rest_of_path, Pair)
            return :tree => direction => rest_of_path[2]
        else
            return :tree => direction
        end
    end
end

@gen function jump_proposal(trace, xs)
    # Choose node to change.
    path ~ random_node_path(get_retval(trace))
    # Sample new expression for that node.
    new_subtree ~ sample_expression()
    return path
end

function involution(trace, forward_choices, path_to_subtree, proposal_args)
    # Need to return a new trace, backward_choices, and a weight.
    backward_choices = choicemap()

    # In the backward direction, the `random_node_path` function should
    # make all the same choices, so that the same exact node is reached
    # for resimulation.
    set_submap!(backward_choices, :path, get_submap(forward_choices, :path))
    
    # But in the backward direction, the `:new_subtree` generation should
    # produce the *existing* subtree.
    set_submap!(backward_choices, :new_subtree, get_submap(get_choices(trace), path_to_subtree))
    
    # The new trace should be just like the old one, but we are updating everything
    # about the new subtree.
    new_trace_choices = choicemap()
    set_submap!(new_trace_choices, path_to_subtree, get_submap(forward_choices, :new_subtree))
    
    # Run update and get the new weight.
    new_trace, weight, = update(trace, get_args(trace), (UnknownChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end

function custom_update_inv(trace, xs)
    trace, accepted = mh(trace, jump_proposal, (xs,), involution)
    trace, = mh(trace, Gen.select(:std))
    trace
end

function do_inference(model, xs, ys, num_iters)
    observations = choicemap()
    # for (i, y) in enumerate(ys)
    #     observations[(:y, i)] = y
    # end
    observations[:y] = ys

    trace, = generate(model, (xs,), observations)
    traces = []
    @showprogress 1 "Sampling..." for i in 1:num_iters
        trace = custom_update_inv(trace, xs)
        push!(traces, trace)
    end
    traces
end

add_path(path::Symbol, to_add::Symbol) = path => to_add
add_path(path::Pair, to_add::Symbol) = path[1] => add_path(path[2], to_add)

function tree_to_dict!(res, prefix, path, trace)
    if isa(trace[path], Identity)
        res["$(prefix)_rule"] = 0
        res["$(prefix)_dummy"] = trace[path].a
    elseif isa(trace[path], Sine)
        res["$(prefix)_rule"] = 1
        res["$(prefix)_a"] = trace[path].a
        tree_to_dict!(res, "$(prefix)_sin_", add_path(path, :subexpr), trace)
    elseif isa(trace[path], Add)
        res["$(prefix)_rule"] = 2
        res["$(prefix)_a"] = trace[path].a
        res["$(prefix)_b"] = trace[path].b
        tree_to_dict!(res, "$(prefix)_plus1_", add_path(path, :left), trace)
        tree_to_dict!(res, "$(prefix)_plus2_", add_path(path, :right), trace)
    end
end

function trace_to_dict(trace)
    res = Dict()
    tree_to_dict!(res, "", :tree, trace)
    res["std"] = trace[:std]
    res
end

function main(data_dir, fname; num_samples=100)
    data_path = joinpath(data_dir, "data.pickle")
    if !isfile(data_path)
        println("No data file found for $data_dir")
    end
    (
        xs,
        ys,
        _, 
        _
    ) = Pickle.Torch.THload(data_path)
    # xs = rand(Uniform(-5, 5), 150)
    # mean_ys = 2 .* sin.(2.0 .* xs) .- xs
    # noise = rand(Normal(0, 0.1), 150)
    # ys = mean_ys .+ noise
    traces = do_inference(model, xs, ys, num_samples)
    trace_dicts = map(trace_to_dict, traces)

    fname = joinpath("results", "fun_ind", "$fname.pickle")
    store(fname, Dict("traces" => trace_dicts))

    trace_dicts, traces
end

function all_main(dir_name; num_samples=100)
    items = readdir(dir_name)
    directories = filter(x -> isdir(joinpath(dir_name, x)), items)
    for dname in directories
        fname = "traces_$(dname)"
        if isfile(joinpath("results", "fun_ind", "$fname.pickle"))
            # Skip already computed results.
            continue
        end
        main(
            joinpath(dir_name, dname), 
            fname;
            num_samples=num_samples,
        )
    end
end