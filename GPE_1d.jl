using FFTW, Plots, LinearAlgebra, Term.Progress, Statistics



function nonlinear_step(ψ,potential,κ1,dt,ϵ,x,λ)
    return exp.(-1im*(κ1.*abs2.(ψ) .+ potential)*dt/(2ϵ))
    #kappa(x,κ1,λ)
end
function imaginary_nonlinear_step(ψ,potential,κ1,dt,ϵ,x,λ)
    return exp.(-(κ1.*abs2.(ψ) .+ potential) * dt/(2ϵ))
end

function evolution_loop(ψ,potential,μ,κ1,dt,ϵ,x,λ)
    ψ .*= nonlinear_step(ψ,potential,κ1,dt,ϵ,x,λ)
    ψf = fft(ψ)
    ψf .*= exp.(-1im *ϵ ./2 .*μ .^2 .*dt)
    ψff = ifft(ψf)
    ψff .*= nonlinear_step(ψff,potential,κ1,dt,ϵ,x,λ)
    return ψff
end
function imaginary_evolution_loop(ψ,potential,μ,κ1,dt,ϵ,x,λ)
    ψ .*= imaginary_nonlinear_step(ψ,potential,κ1,dt,ϵ,x,λ)
    ψf = fft(ψ)
    ψf .*= exp.(-ϵ ./2 .*μ .^2 .*dt)
    ψff = ifft(ψf)
    ψff .*= imaginary_nonlinear_step(ψff,potential,κ1,dt,ϵ,x,λ)
    return ψff
end

function average(ψ,x,dx)
    prob_density = abs2.(ψ)
    x_avg = sum(x.*prob_density)*dx
    x2_avg = sum(x.^2 .*prob_density)*dx
    return sqrt(x2_avg - x_avg^2)
end

function x_center(ψ,x,dx)
    prob_density = abs2.(ψ)
    x_avg = sum(x.*prob_density)*dx
    return x_avg
end

function box_create(a)
    if -2 < a < 2
        return 0
    else
        return Inf
    end
end

function kappa(x,κ0,λ)
    return κ0 ./2 .*(1 .+cos.(2π .*x ./λ))
end

function nx(x,N_part,κ1,ω)
    temp = ((3/(4*sqrt(2))*N_part*κ1*ω)^(2/3).-1/2*ω^2*x .^2)/κ1
    return temp
end

function box_potential(x,L,V0,steep)
    return V0*(1 ./(1 .+exp.(-(x.-L)/steep)) .+ 1 ./(1 .+exp.((x .+L)/steep)))
end

function real_part()
    #initial parameters
    ϵ = 1.0
    κ1 = 5.0
    ω = 1.0
    x0 = 0
    x0_pot = 0
    N_part = 100

    M = 2^9  # Number of spatial points
    a = -8.0 #spatial minimum
    b = 8.0 #spatial maximum
    L = b - a
    dx = (b-a) / (M)
    dt = 0.0001  # Time step
    Nt = 200000  # Number of time steps
    t = LinRange(0,Nt*dt,Nt)
    x = LinRange(a,b,M)
    μ = 2π/L*vcat(0:(M/2-1),-M/2:-1)

    ψ = zeros(ComplexF64,M,Nt)
    ψ[:,1] .= exp.(-(x .-x0).^2) # Initial Wavefunction
    ψ[:,1] ./= sqrt(sum(abs2.(ψ))*dx) # Normalization

    σ = zeros(Float64,Nt)
    σ[1] = average(ψ[:,1])

    X = zeros(Float64,Nt)
    X[1] = x_center(ψ[:,1])

    #progress bar
    pbar = ProgressBar()
    job = addjob!(pbar;N=Nt)
    start!(pbar)
    update!(job)

    #Computation loop
    for i in 2:Nt
        ψ[:,i] = evolution_loop(ψ[:,i-1],ω^2 * x.^2 /2)
        σ[i] = average(ψ[:,i])
        X[i] = x_center(ψ[:,i])
        update!(job)
        render(pbar)
    end

    maxima_index = findall(i -> σ[i] > σ[i-1] && σ[i] > σ[i+1], 2:Nt-1)
    pulsation = 2π/(t[maxima_index[2]]-t[maxima_index[1]])

    maxima_index = findall(i -> X[i] > X[i-1] && X[i] > X[i+1], 2:Nt-1)
    pulsation_X = 2π/(t[maxima_index[2]]-t[maxima_index[1]])

    #Plot loop
    job2 = addjob!(pbar;N=100)
    l = @layout [a{0.6h};b ;d]
    @gif for i in range(1,Nt,length=100)
        update!(job2)
        render(pbar)
        i_ = round(Int,i)
        plt1 = plot(x,abs.(ψ[:,i_]),label="ψ")
        #plot!(x,imag(ψ[:,i_]),label="imag_ψ")
        plot!(x,x.^2 ./2,label="V")
        ylims!(0,1)
        title!("GPE in 1d for $M points, Nt=$Nt and dt=$dt,T=$(round(i * dt, digits=2))",titlefont=font(10))
        plt2 = plot(t,σ,label="σ",title="pulsation = $(round(pulsation,digits = 3))")
        plt3 = plot(t,X,title="pulsation=$pulsation_X",label="x_moyen")
        plot(plt1,plt2,plt3,layout = l)
    end
end

function imaginary_part()
    #initial parameters
    ϵ = 1.0
    κ1 = 10.0
    ω = 1.0
    x0 = 0
    x0_pot = 0
    N_part = 100

    M = 2^9  # Number of spatial points
    a = -15.0 #spatial minimum
    b = 15.0 #spatial maximum
    L = b - a
    dx = (b-a) / (M)
    dt = 0.00001  # Time step
    Nt = 400000  # Number of time steps
    t = LinRange(0,Nt*dt,Nt)
    x = LinRange(a,b,M)
    μ = 2π/L*vcat(0:(M/2-1),-M/2:-1)

    V0 = 1e9
    steep = 0.05
    λ = steep*10
    size_of_pot = λ*10
    box = box_potential(x,size_of_pot,V0,steep)

    ψ = zeros(ComplexF64,M,Nt)
    ψ[:,1].= exp.(-(x .-x0).^2) # Initial Wavefunction
    ψ[:,1] ./= sqrt(sum(abs2.(ψ))*dx) / sqrt(N_part) # Normalization
    #progress bar
    pbar = ProgressBar()
    job = addjob!(pbar;N=Nt)
    start!(pbar)
    update!(job)

    #Computation loop
    for i in 2:Nt
        ψ[:,i] = imaginary_evolution_loop(ψ[:,i-1],ω^2*x.^2/2,μ,κ1,dt,ϵ,x,λ)
        ψ[:,i] ./= sqrt(sum(abs2.(ψ[:,i]))*dx) / sqrt(N_part)
        update!(job)
        render(pbar)
        if i == round(Int,3Nt/4)
            ψ[:,i] = abs.(ψ[:,i])
        end
    end

    #gif
    # job2 = addjob!(pbar;N=100)
    # @gif for i in range(1,Nt,length=100)
    #     update!(job2)
    #     render(pbar)
    #     i_ = round(Int,i)
    #     plt1 = plot(x,abs2.(ψ[:,i_]),title="imaginary time evolution T=$(trunc(i * dt, digits=2))",label="ψ")
    #     plot!(x,nx(x,N_part,κ1,ω),label="imag")
    #     plt2 = plot(x,angle.(ψ[:,i_]))
    #     plot(plt1,plt2)
    # end

    #stationnary plot
    #l = @layout [a{0.6h};b]
    # plt1 = plot(x,abs2.(ψ[:,end]),xlabel="x",ylabel="density",title="imaginary time evolution",label="ψ",ylims=(0,maximum(abs2.(ψ[:,end]))))
    # plot!(x,ω^2*x.^2/2,label="potential")
    # plot!(x,nx(x,N_part,κ1,ω),label="Thomas-Fermi")
    plt2 = plot(x,angle.(ψ[:,end]),label="phase of ψ",xlabel="x",title="Phase of ψ after imaginary time",ylabel="phase")
    # plot!(x,angle.(ψ[:,round(Int,Nt/3)]),label="angle at t = $(round(Int,Nt/3*dt))")
    # plot!(x,angle.(ψ[:,round(Int,2*Nt/3)]),label="angle at t = $(round(Int,2*Nt/3*dt))")
    #plot(plt1,plt2,layout=l)
    savefig("imaginary_plot_harmonic_theory_angle.png")
end

function main_gif()
    ψ = exp.(-(x .-x0).^2)
    ψ ./= sqrt(sum(abs2.(ψ))*dx) / sqrt(N_part)

    #progress bar
    pbar = ProgressBar()
    job = addjob!(pbar;N=Nt)
    start!(pbar)

    ω = 1
    #Computation loop
    for i in 1:Int(Nt)
        ψ = imaginary_evolution_loop(ψ,ω^2 * x.^2 /2)
        ψ ./= sqrt(sum(abs2.(ψ))*dx) / sqrt(N_part)
        update!(job)
        render(pbar)
    end

    σ = zeros(Float64,1)
    σ[1] = average(ψ)

    X = zeros(Float64,1)
    X[1] = x_center(ψ)

    variation = 0
    ω *= variation+1
    α = Int(Nt/100)
    l = @layout [a{0.6h};b;c]
    job2 = addjob!(pbar;N=Nt)
    @gif for i in 1:100
        for j in 1:α
            ψ = evolution_loop(ψ,ω^2 * (x .-0.2).^2 /2)
            update!(job2)
            render(pbar)
            σ = append!(σ,average(ψ))
            X = append!(X,x_center(ψ))
        end
        plt1 = plot(x,abs.(ψ),title="GPE with imaginary then real time\npotential variation = $(variation*100)%\nT = $(round(i*α*dt;digits=1))",label="ψ")
        plot!(x,x.^2/2,label="potential")
        ylims!(0,6)
        
        maxima_index = findall(j -> σ[j] < σ[j-1] && σ[j] < σ[j+1] && j<length(σ), 2:length(σ)-1)
        pulsation = 0
        if length(maxima_index) > 0
            pulsation = 2π/(t[maxima_index[1]])
        end
        plt2 = plot(t[1:length(σ)-1],σ[2:end],label="spread σ",title="pulsation = $pulsation",legend=:bottomright)

        maxima_index = findall(j -> X[j] < X[j-1] && X[j] < X[j+1] && j<length(X), 2:length(X)-1)
        pulsation_X = 0
        if length(maxima_index) > 0
            pulsation_X = 2π/(t[maxima_index[1]])
        end
        plt3 = plot(t[1:length(X)-1],X[2:end],label="x_moyen",title="pulsation = $pulsation_X",legend=:bottomright)

        plot(plt1,plt2,plt3,layout = l)
    end
end

function main_plt()
    #initial parameters
    ϵ = 1.0
    κ1 = 10.0
    ω = 0.1
    x0 = 0
    size_of_pot = 5
    λ = size_of_pot/10
    steep = λ/10
    x0_pot = 0
    N_part = 100

    M = 2^9  # Number of spatial points
    a = -15.0 #spatial minimum
    b = -a #spatial maximum
    L = b - a
    dx = (b-a) / (M)
    dt = 0.0001  # Time step
    Nt = 4000000  # Number of time steps
    t = LinRange(0,Nt*dt,Nt)
    x = LinRange(a,b,M)
    μ = 2π/L*vcat(0:(M/2-1),-M/2:-1)

    V0 = 1e9
    box = box_potential(x,size_of_pot,V0,steep)

    ψ_im = exp.(-(x .-x0).^2)
    ψ_im ./= sqrt(sum(abs2.(ψ_im))*dx)
    println(sqrt(sum(abs2.(ψ_im))*dx))

    #progress bar
    pbar = ProgressBar()
    job = addjob!(pbar;N=Int(Nt))
    start!(pbar)

    #Computation loop
    for i in 1:Int(400000)
        ψ_im = imaginary_evolution_loop(ψ_im,box,μ,κ1,dt/10,ϵ,x,λ)
        ψ_im ./= sqrt(sum(abs2.(ψ_im))*dx)
        update!(job)
        render(pbar)
    end
    ψ_im = abs.(ψ_im)
    println(sqrt(sum(abs2.(ψ_im))*dx))
    ψ = zeros(ComplexF64,M,Nt)
    ψ[:,1] = ψ_im

    σ = zeros(Float64,1)
    σ[1] = average(ψ[:,1],x,dx)

    X = zeros(Float64,1)
    X[1] = x_center(ψ[:,1],x,dx)

    variation = 0
    ω /= variation+1
    job2 = addjob!(pbar;N=Nt)
    update!(job2)
    for i in 2:Nt
        ψ[:,i] = evolution_loop(ψ[:,i-1],box,μ,κ1,dt,ϵ,x,λ)
        update!(job2)
        render(pbar)
        σ = append!(σ,average(ψ[:,i],x,dx))
        X = append!(X,x_center(ψ[:,i],x,dx))
    end
    println(sqrt(sum(abs2.(ψ[:,end]))*dx))
    stop!(pbar)
    plt1 = heatmap(t,x,abs.(ψ),c =:jet1,title="GPE with imaginary then real time\npotential variation = $(variation*100)%\n",label="ψ",xlabel="t",ylabel="x")
    
    maxima_index = findall(j -> σ[j] < σ[j-1] && σ[j] < σ[j+1], 2:length(σ)-1)
    pulsation = 0
    for i in 1:length(maxima_index)-1
        pulsation = 2π/(t[maxima_index[i+1]] - t[maxima_index[i]])
        println("temps minimum = $(t[maxima_index[i]])")
    end
    plt2 = plot(t,σ,label="spread σ",title="pulsation = $(round(pulsation,digits = 3))",ylims = (0,maximum(σ)))
    
    maxima_index = findall(j -> X[j] < X[j-1] && X[j] < X[j+1], 2:length(X)-1)
    pulsation_X = 0
    for i in 1:length(maxima_index)-1
        pulsation_X = 2π/(t[maxima_index[i+1]] - t[maxima_index[i]])
    end
    plt3 = plot(t,X,label="x_moyen",title="pulsation = $(round(pulsation_X,digits=3))")
    
    println("disaplying plot...")
    l = @layout [a{0.6h};b;c]
    plt = plot(plt1,plt2,plt3,layout=l,size=(1200,800))
    display(plt)
    return 0
end

imaginary_part()