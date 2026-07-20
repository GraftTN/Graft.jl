using Graft
using Graft.TestUtils
using LinearAlgebra
using Printf

function lifted_operator(site, name, op, topo, phys, nmax)
    observable = OpSum() + Term(1.0, SiteOp(site, name, op))
    dressed, _, _ = ppdress(observable, topo, phys;
                            nmax, boson_sites=[:ph])
    return only(only(collect(dressed)).ops).op
end

function gauss_legendre(n, beta)
    offdiag = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    rule = eigen(SymTridiagonal(zeros(n), offdiag))
    return beta .* (rule.values .+ 1) ./ 2,
           beta .* abs2.(rule.vectors[1, :])
end

function main()
    beta, U, mu, V = 10.0, 4.0, 2.0, 0.35
    omega, g, nmax = 0.1, sqrt(0.02), 4

    S = spin_ops()
    E = (; C=S.Sp, Cd=S.Sm, N=S.N, I=S.I, P=S.P)
    B = boson_ops(nmax)
    fermions = [:d_up, :bath_up, :d_dn, :bath_dn]
    sites = [fermions; :ph]
    topo = TreeTopology(last(sites),
        [sites[i + 1] => sites[i] for i in (length(sites) - 1):-1:1])
    phys = Dict(site => E.P for site in fermions)
    phys[:ph] = B.P

    H = OpSum() +
        Term(U, SiteOp(:d_up, :N, E.N), SiteOp(:d_dn, :N, E.N)) +
        Term(-mu, SiteOp(:d_up, :N, E.N)) +
        Term(-mu, SiteOp(:d_dn, :N, E.N)) +
        Term(V, SiteOp(:d_up, :Cd, E.Cd), SiteOp(:bath_up, :C, E.C)) +
        Term(V, SiteOp(:d_up, :C, E.C), SiteOp(:bath_up, :Cd, E.Cd)) +
        Term(V, SiteOp(:d_dn, :Cd, E.Cd), SiteOp(:bath_dn, :C, E.C)) +
        Term(V, SiteOp(:d_dn, :C, E.C), SiteOp(:bath_dn, :Cd, E.Cd)) +
        Term(omega, SiteOp(:ph, :N, B.N)) +
        Term(g, SiteOp(:d_up, :N, E.N), SiteOp(:ph, :X, B.X)) +
        Term(g, SiteOp(:d_dn, :N, E.N), SiteOp(:ph, :X, B.X)) +
        Term(-g, SiteOp(:d_up, :I, E.I), SiteOp(:ph, :X, B.X))

    Hp, topop, physp_raw = ppdress(H, topo, phys; nmax, boson_sites=[:ph])
    Space = typeof(first(values(physp_raw)))
    physp = Dict{Symbol,Space}(site => space for (site, space) in physp_raw)
    problem = purification_problem(Hp, topop, physp;
                                   hermitian=true,
                                   pp_pairs=Dict(:ph => :ph_B1))

    taus, weights = gauss_legendre(12, beta)
    evolver = TDVP2(trunc=TruncationScheme(maxdim=16, atol=1e-10),
                    krylovdim=24, tol=1e-10, verbose=false)
    trajectory = thermalize(Purified(), problem, beta;
        evolver, nsteps=20,
        save_betas=sort(unique(vcat(beta .- taus, [beta]))))

    Nup = lifted_operator(:d_up, :N, E.N, topo, phys, nmax)
    Ndn = lifted_operator(:d_dn, :N, E.N, topo, phys, nmax)
    upup = thermal_correlator(
        Purified(), problem, :d_up => Nup, :d_up => Nup, beta, taus;
        evolver, trajectory, prop_nsteps=20, connected=true)
    updn = thermal_correlator(
        Purified(), problem, :d_up => Nup, :d_dn => Ndn, beta, taus;
        evolver, trajectory, prop_nsteps=20, connected=true)
    chi = 2 .* real.(upup.values .+ updn.values)

    Hd = dense_hamiltonian(H, topo, phys)
    Nupd = dense_hamiltonian(
        OpSum() + Term(1.0, SiteOp(:d_up, :N, E.N)), topo, phys)
    Ndnd = dense_hamiltonian(
        OpSum() + Term(1.0, SiteOp(:d_dn, :N, E.N)), topo, phys)
    nup = real(exact_thermal_expect(Hd, Nupd, beta))
    ndn = real(exact_thermal_expect(Hd, Ndnd, beta))
    chi_ed = 2 .* real.(
        exact_thermal_correlator(Hd, Nupd, Nupd, beta, taus) .- nup^2 .+
        exact_thermal_correlator(Hd, Nupd, Ndnd, beta, taus) .- nup * ndn)

    println("Anderson-Holstein: g*(n_up+n_dn-1)*(b+bdagger)")
    @printf("beta=%.1f, nmax=%d, <n_up>_TTNS=%.8f\n",
            beta, nmax, real(upup.metadata.Abar))
    @printf("chi_N(i nu_0): TTNS=%.12f, ED=%.12f\n",
            dot(weights, chi), dot(weights, chi_ed))
    println("tau        chi_TTNS       chi_ED")
    for i in eachindex(taus)
        @printf("%4.1f   % .10f   % .10f\n", taus[i], chi[i], chi_ed[i])
    end
end

main()
