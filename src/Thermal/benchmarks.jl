"""
    FiniteModeBenchmarkCell

One complete P4 Anderson-Holstein benchmark result. A cell stores the model,
representation, all required observables, and the numerical provenance needed
to separate time-step, bond, boson-cutoff, and PP-LBO errors.
"""
struct FiniteModeBenchmarkCell
    label::Symbol
    action::FiniteModeAction
    method::Symbol
    representation::Symbol
    nmax::Int
    data::Vector{ThermalBenchmarkDatum}
    max_bond_dimension::Int
    truncation::NamedTuple
    propagation_grid::Vector{Float64}
    wall_time::Float64
    function FiniteModeBenchmarkCell(
            label::Symbol,
            action::FiniteModeAction,
            method::Symbol,
            representation::Symbol,
            nmax::Integer,
            data;
            max_bond_dimension::Integer=0,
            truncation::NamedTuple=(;),
            propagation_grid=Float64[],
            wall_time::Real=0)
        method in (:ed, :graft, :ctseg) ||
            throw(ArgumentError("benchmark method must be :ed, :graft, or :ctseg"))
        representation in (:plain, :pp_untruncated, :pp_lbo, :integrated) ||
            throw(ArgumentError("unsupported benchmark representation"))
        nmax >= 0 || throw(ArgumentError("nmax must be nonnegative"))
        max_bond_dimension >= 0 ||
            throw(ArgumentError("max_bond_dimension must be nonnegative"))
        time = Float64(wall_time)
        isfinite(time) && time >= 0 ||
            throw(ArgumentError("wall_time must be finite and nonnegative"))
        grid = Float64.(collect(propagation_grid))
        all(isfinite, grid) && issorted(grid) ||
            throw(ArgumentError("propagation_grid must be finite and sorted"))
        values = ThermalBenchmarkDatum[datum for datum in data]
        _check_benchmark_data(action, values)
        return new(
            label, action, method, representation, Int(nmax), values,
            Int(max_bond_dimension), truncation, grid, time)
    end
end

function _check_benchmark_data(action, data)
    isempty(data) && throw(ArgumentError("benchmark cell has no data"))
    seen = Set{Tuple{Symbol,Symbol,UInt64}}()
    for datum in data
        key = (
            datum.observable, datum.axis,
            reinterpret(UInt64, datum.coordinate))
        key in seen &&
            throw(ArgumentError("duplicate benchmark datum $(key[1:2]) at $(datum.coordinate)"))
        push!(seen, key)
    end

    required_scalars = Symbol[:logZ, :density, :boson_occupation]
    action.orbital_convention === :spinful &&
        push!(required_scalars, :double_occupancy)
    for observable in required_scalars
        _has_observable(data, observable, :scalar) ||
            throw(ArgumentError("benchmark cell is missing scalar $observable"))
    end
    for (observable, axis) in (
            (:Gtau, :tau),
            (:chi_nn, :tau),
            (:Giw, :fermionic_iw),
            (:chi_nn_iv, :bosonic_iv))
        _has_observable(data, observable, axis) ||
            throw(ArgumentError("benchmark cell is missing $observable on $axis"))
    end
    return nothing
end

_has_observable(data, observable, axis) =
    any(datum -> datum.observable === observable && datum.axis === axis, data)

struct BosonCutoffReport
    converged::Bool
    rows::Vector{NamedTuple}
    final_pair::Tuple{Int,Int}
end

"""
    assess_boson_cutoff(cells; target_uncertainty, coordinate_atol=0)

Compare successive `nmax` cells. The overall gate is the final cutoff pair;
earlier pairs remain in `rows` to expose the convergence trend.
"""
function assess_boson_cutoff(cells;
                             target_uncertainty,
                             coordinate_atol::Real=0)
    values = sort(
        FiniteModeBenchmarkCell[cell for cell in cells];
        by=cell -> cell.nmax)
    length(values) >= 2 ||
        throw(ArgumentError("boson-cutoff scan needs at least two cells"))
    allunique(cell.nmax for cell in values) ||
        throw(ArgumentError("boson-cutoff scan has duplicate nmax values"))
    first_cell = first(values)
    for cell in values[2:end]
        cell.label == first_cell.label ||
            throw(ArgumentError("cutoff scan mixes benchmark labels"))
        finite_mode_hash(cell.action) == finite_mode_hash(first_cell.action) ||
            throw(ArgumentError("cutoff scan mixes finite-mode actions"))
        cell.method == first_cell.method ||
            throw(ArgumentError("cutoff scan mixes benchmark methods"))
        cell.representation == first_cell.representation ||
            throw(ArgumentError("cutoff scan mixes representations"))
    end
    atol = Float64(coordinate_atol)
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("coordinate_atol must be finite and nonnegative"))

    rows = NamedTuple[]
    for (low, high) in zip(values[1:end - 1], values[2:end])
        append!(rows, _compare_cells(
            low, high;
            allowance=observable ->
                _observable_tolerance(target_uncertainty, observable),
            coordinate_atol=atol,
            comparison=:boson_cutoff))
    end
    final_pair = (values[end - 1].nmax, values[end].nmax)
    final_rows = filter(
        row -> row.low_nmax == final_pair[1] &&
            row.high_nmax == final_pair[2],
        rows)
    return BosonCutoffReport(
        all(row -> row.passed, final_rows), rows, final_pair)
end

struct RepresentationComparison
    passed::Bool
    rows::Vector{NamedTuple}
    reference_representation::Symbol
    candidate_representation::Symbol
end

"""
    compare_representations(reference, candidate; tolerance,
                            coordinate_atol=0)

Compare plain, untruncated PP, or truncated PP cells. The candidate's
per-datum `deterministic_error` and `cutoff_error` are not silently enlarged;
the explicit LBO allowance is carried in `candidate.lbo_error` for `:pp_lbo`
cells and is reported separately in every row.
"""
function compare_representations(
        reference::FiniteModeBenchmarkCell,
        candidate::FiniteModeBenchmarkCell;
        tolerance,
        coordinate_atol::Real=0)
    reference.label == candidate.label ||
        throw(ArgumentError("representation comparison mixes benchmark labels"))
    finite_mode_hash(reference.action) == finite_mode_hash(candidate.action) ||
        throw(ArgumentError("representation comparison mixes finite-mode actions"))
    reference.nmax == candidate.nmax ||
        throw(ArgumentError("representation comparison requires equal nmax"))
    reference.method == candidate.method ||
        throw(ArgumentError("representation comparison requires equal methods"))
    reference.representation == candidate.representation &&
        throw(ArgumentError("representation comparison needs distinct representations"))
    atol = Float64(coordinate_atol)
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("coordinate_atol must be finite and nonnegative"))

    rows = _compare_cells(
        reference, candidate;
        allowance=observable -> _observable_tolerance(tolerance, observable),
        coordinate_atol=atol,
        comparison=:representation)
    return RepresentationComparison(
        all(row -> row.passed, rows), rows,
        reference.representation, candidate.representation)
end

function _compare_cells(low, high; allowance, coordinate_atol, comparison)
    rows = NamedTuple[]
    for right in high.data
        matches = filter(low.data) do left
            left.observable === right.observable &&
                left.axis === right.axis &&
                abs(left.coordinate - right.coordinate) <= coordinate_atol
        end
        isempty(matches) &&
            throw(ArgumentError(
                "missing $(right.observable)/$(right.axis) at $(right.coordinate)"))
        length(matches) == 1 ||
            throw(ArgumentError(
                "ambiguous $(right.observable)/$(right.axis) at $(right.coordinate)"))
        left = only(matches)
        base = allowance(right.observable)
        errors = left.deterministic_error + right.deterministic_error +
            left.cutoff_error + right.cutoff_error
        lbo = comparison === :representation &&
            high.representation === :pp_lbo ? right.lbo_error : 0.0
        threshold = base + errors + lbo
        delta = abs(left.value - right.value)
        push!(rows, (;
            observable=right.observable,
            axis=right.axis,
            coordinate=right.coordinate,
            low_nmax=low.nmax,
            high_nmax=high.nmax,
            reference=left.value,
            candidate=right.value,
            absolute_error=delta,
            target_uncertainty=base,
            deterministic_allowance=errors,
            lbo_allowance=lbo,
            threshold,
            passed=delta <= threshold,
        ))
    end
    return rows
end

function _observable_tolerance(tolerance::Real, observable)
    value = Float64(tolerance)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("benchmark tolerance must be finite and nonnegative"))
    return value
end

function _observable_tolerance(tolerance::AbstractDict, observable)
    haskey(tolerance, observable) ||
        throw(ArgumentError("missing tolerance for observable $observable"))
    return _observable_tolerance(tolerance[observable], observable)
end

"""
    write_finite_mode_benchmark_csv(path, cells)

Write complete P4 cells as a flat CSV. Action parameters, numerical settings,
and decomposed errors are repeated on every observable row.
"""
function write_finite_mode_benchmark_csv(path::AbstractString, cells)
    values = FiniteModeBenchmarkCell[cell for cell in cells]
    isempty(values) && throw(ArgumentError("no benchmark cells to write"))
    header = [
        "label", "action_hash", "method", "representation", "beta", "mu",
        "static_interaction", "n0", "bath_energies", "bath_couplings",
        "boson_frequencies", "boson_couplings", "orbital_convention",
        "density_convention", "nmax", "max_bond_dimension", "truncation",
        "propagation_grid", "wall_time", "observable", "axis", "coordinate",
        "value_re", "value_im", "stderr", "deterministic_error",
        "cutoff_error", "lbo_error",
    ]
    open(path, "w") do io
        println(io, join(header, ','))
        for cell in values
            action = cell.action
            prefix = Any[
                cell.label, finite_mode_hash(action), cell.method,
                cell.representation, action.beta, action.mu,
                action.static_interaction, action.n0,
                join(action.bath_energies, ';'),
                join((_complex_text(value)
                      for value in action.bath_couplings), ';'),
                join(action.boson_frequencies, ';'),
                join(action.boson_couplings, ';'),
                action.orbital_convention, action.density_convention,
                cell.nmax, cell.max_bond_dimension, repr(cell.truncation),
                join(cell.propagation_grid, ';'), cell.wall_time,
            ]
            for datum in cell.data
                row = [
                    prefix;
                    datum.observable;
                    datum.axis;
                    datum.coordinate;
                    real(datum.value);
                    imag(datum.value);
                    datum.stderr;
                    datum.deterministic_error;
                    datum.cutoff_error;
                    datum.lbo_error;
                ]
                println(io, join(_csv_encode.(row), ','))
            end
        end
    end
    return path
end

_complex_text(value) = string(real(value), ':', imag(value))
