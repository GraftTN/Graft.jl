using Test
using Printf

const TEST_VERBOSE = lowercase(get(ENV, "GRAFT_TEST_VERBOSE", "true")) in
    ("1", "true", "yes", "on")

function _graft_boolean_env(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, string(default))))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    throw(ArgumentError(
        "$name must be one of 1/true/yes/on or 0/false/no/off, got $(repr(raw))",
    ))
end

# Expensive diagnostic prefixes and stress permutations are opt-in; the
# default tier retains public action/ED/topology/interface contracts.
const GRAFT_EXTENDED_TESTS = _graft_boolean_env("GRAFT_EXTENDED_TESTS", false)

function _graft_positive_env_int(name::String, default::Int)
    raw = get(ENV, name, string(default))
    value = tryparse(Int, raw)
    value === nothing && throw(ArgumentError("$name must be an integer, got $(repr(raw))"))
    value >= 1 || throw(ArgumentError("$name must be at least 1, got $value"))
    return value
end

const GRAFT_TEST_SHARD_COUNT = _graft_positive_env_int("GRAFT_TEST_SHARD_COUNT", 1)
const GRAFT_TEST_SHARD_INDEX = _graft_positive_env_int("GRAFT_TEST_SHARD_INDEX", 1)
GRAFT_TEST_SHARD_INDEX <= GRAFT_TEST_SHARD_COUNT || throw(ArgumentError(
    "GRAFT_TEST_SHARD_INDEX ($GRAFT_TEST_SHARD_INDEX) must not exceed " *
    "GRAFT_TEST_SHARD_COUNT ($GRAFT_TEST_SHARD_COUNT)"))
println("[shard] index=$GRAFT_TEST_SHARD_INDEX count=$GRAFT_TEST_SHARD_COUNT")
println("[tier] extended=$GRAFT_EXTENDED_TESTS")

const _GRAFT_TEST_STAGE_ORDINAL = Ref(0)
const _GRAFT_TEST_STARTED = time_ns()

mutable struct _GraftTestTotals
    passed::Int
    failed::Int
    errors::Int
    broken::Int
    stages::Int
end

const _GRAFT_TEST_TOTALS = _GraftTestTotals(0, 0, 0, 0, 0)

function _graft_next_test_stage()
    ordinal = (_GRAFT_TEST_STAGE_ORDINAL[] += 1)
    selected = mod1(ordinal, GRAFT_TEST_SHARD_COUNT) == GRAFT_TEST_SHARD_INDEX
    return (; ordinal, selected)
end

function _graft_report_test_stage(testset::Test.DefaultTestSet, elapsed::Float64)
    counts = Test.get_test_counts(testset)
    passed = counts.passes + counts.cumulative_passes
    failed = counts.fails + counts.cumulative_fails
    errors = counts.errors + counts.cumulative_errors
    broken = counts.broken + counts.cumulative_broken
    total = passed + failed + errors + broken
    elapsed_text = @sprintf "%.3fs" elapsed
    _GRAFT_TEST_TOTALS.passed += passed
    _GRAFT_TEST_TOTALS.failed += failed
    _GRAFT_TEST_TOTALS.errors += errors
    _GRAFT_TEST_TOTALS.broken += broken
    _GRAFT_TEST_TOTALS.stages += 1
    println("[stage] $(testset.description) passed=$passed failed=$failed " *
            "errors=$errors broken=$broken total=$total elapsed=$elapsed_text")
    return testset
end

function _graft_report_test_total()
    totals = _GRAFT_TEST_TOTALS
    total = totals.passed + totals.failed + totals.errors + totals.broken
    elapsed = (time_ns() - _GRAFT_TEST_STARTED) / 1.0e9
    elapsed_text = @sprintf "%.3fs" elapsed
    println("[total] passed=$(totals.passed) failed=$(totals.failed) " *
            "errors=$(totals.errors) broken=$(totals.broken) total=$total " *
            "stages=$(totals.stages) elapsed=$elapsed_text")
    return nothing
end

"""Run a top-level testset when its global round-robin shard is selected."""
macro graft_testset(name, body)
    return quote
        local graft_stage = _graft_next_test_stage()
        if graft_stage.selected
            local graft_started = time_ns()
            local graft_testset = Test.@testset $name $body
            _graft_report_test_stage(graft_testset, (time_ns() - graft_started) / 1.0e9)
        end
    end
end

"""
Run an opt-in top-level testset when its global round-robin shard is selected.

The three-argument form accepts a `force` condition for focused invocations
that need one related extended stage without enabling the full extended tier.
The stage ordinal is consumed regardless of the tier/force decision.
"""
macro graft_extended_testset(args...)
    force, name, body = if length(args) == 2
        (false, args[1], args[2])
    elseif length(args) == 3
        (args[1], args[2], args[3])
    else
        throw(ArgumentError(
            "@graft_extended_testset expects (name, body) or (force, name, body)",
        ))
    end
    return quote
        local graft_stage = _graft_next_test_stage()
        if (GRAFT_EXTENDED_TESTS || $force) && graft_stage.selected
            local graft_started = time_ns()
            local graft_testset = Test.@testset $name $body
            _graft_report_test_stage(graft_testset, (time_ns() - graft_started) / 1.0e9)
        end
    end
end
