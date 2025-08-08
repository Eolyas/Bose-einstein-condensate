using FFTW, Plots, LinearAlgebra, Term.Progress

# Nonlinear step in 2D
function nonlinear_step(ψ,potential_x,potential_y,κ1,dt,ϵ)
    temp = exp.(-1im .* (κ1 .* abs2.(ψ) .+ (potential_x .+ potential_y)) .* dt ./ 2 ./ ϵ)
    return temp
end
function imaginary_nonlinear_step(ψ,potential_x,potential_y,κ1,dt,ϵ)
    return exp.(-(κ1 .* abs2.(ψ) .+ (potential_x .+ potential_y)) .* dt ./ 2 ./ ϵ)
end

function evolution_loop(ψ,potential_x,potential_y,μ,κ1,dt,ϵ)
    ψ .*= nonlinear_step(ψ,potential_x,potential_y,κ1,dt,ϵ)
    ψf = fft(ψ)
    ψf .*= exp.(-1im*ϵ^2/2*(μ.^2 .+ μ'.^2)*dt)
    ψff = ifft(ψf)
    ψff .*= nonlinear_step(ψff,potential_x,potential_y,κ1,dt,ϵ)
    return ψff
end
function imaginary_evolution_loop(ψ,potential_x,potential_y,μ,κ1,dt,ϵ)
    ψ .*= imaginary_nonlinear_step(ψ,potential_x,potential_y,κ1,dt,ϵ)
    ψf = fft(ψ)
    ψf .*= exp.(-ϵ ./2 .*(μ.^2 .+ μ'.^2) .*dt)
    ψff = ifft(ψf)
    ψff .*= imaginary_nonlinear_step(ψff,potential_x,potential_y,κ1,dt,ϵ)
    return ψff
end

function average(ψ,x,dx)
    prob_density = abs2.(ψ)
    x_avg = sum(x.*prob_density)*dx^2
    y_avg = sum(x'.*prob_density)*dx^2
    x2_avg = sum(x.^2 .*prob_density)*dx^2
    y2_avg = sum(x.^2 .*prob_density)*dx^2
    σx = sqrt(x2_avg - x_avg^2)
    σy = sqrt(y2_avg - y_avg^2)
    σ = sqrt(σx^2 + σy^2)
    return [σx; σy; σ]
end

function x_center(ψ,x,dx)
    prob_density = abs2.(ψ)
    x_avg = sum(x.*prob_density)*dx^2
    return x_avg
end
function y_center(ψ,x,dx)
    prob_density = abs2.(ψ)
    x_avg = sum(x'.*prob_density)*dx^2
    return x_avg
end

# Main function to run the simulation
function main()
    # Initial parameters
    ϵ = 1.0
    ωy = 1.0
    ωx = 1.0
    κ1 = 10.0
    x0 = 0
    y0 = 0
    N_part = 100

    h = 1/(2^5)
    a = -10.0 # Spatial minimum for both x and y
    b = -a # Spatial maximum for both x and y
    L = b - a
    M = Int(L/h)
    dx = (b-a)/M
    dt = 0.001  # Time step
    Nt = 8000  # Number of time steps
    t = LinRange(0,Nt*dt,Nt)
    x = LinRange(a, b, M)
    potential_x = ωx^2 * x.^2 /2
    potential_y = ωy^2 * x'.^2 /2

    μ = 2 * π / L * vcat(0:(M/2-1),-M/2:-1)

    # Initialize wavefunction ψ
    ψ = zeros(ComplexF64, M, M)

    ψ .= exp.(-(x .- x0).^2 .- (x' .- y0).^2)
    #ψ .= 1/sqrt(π*ϵ) * exp.((-x.^2 .- x'.^2)/(2*ϵ))  # Initial condition
    #ψ .= γ^(1/4)/sqrt(π*ϵ) * exp.((-x.^2 .- γ .* x'.^2)/(2*ϵ)) * exp.(1im .*cosh.(sqrt.(x .^2 .+ 2 .* x'.^2)) ./ϵ)
    ψ ./= sqrt(sum(abs2.(ψ))*dx^2) /sqrt(N_part)  # Normalize the initial wavefunction

    # Progress bar
    pbar = ProgressBar()
    job = addjob!(pbar; N = Nt)
    start!(pbar)

    println(sqrt(sum(abs2.(ψ))*dx^2) /sqrt(N_part))
    for i in 1:Nt
        ψ = imaginary_evolution_loop(ψ,potential_x,potential_y,μ,κ1,dt,ϵ)
        ψ ./= sqrt(sum(abs2.(ψ))*dx^2) / sqrt(N_part)
        update!(job)
        render(pbar)
        if i == round(Int,Nt*2/3)
            ψ = abs.(ψ)
        end
    end
    ψ = abs.(ψ)
    println(sqrt(sum(abs2.(ψ))*dx^2) /sqrt(N_part))

    σ = zeros(Float64,3,1)
    σ = hcat(σ, average(ψ,x,dx))

    X = zeros(Float64,1)
    X[1] = x_center(ψ,x,dx)
    Y = zeros(Float64,1)
    Y[1] = y_center(ψ,x,dx)

    #parameters for plot/gif
    potential_x = (ωx*1.05)^2 * x.^2 /2
    potential_y = (ωy*1.05)^2 * x.^2 /2
    dim = b
    Nf = 100
    l = @layout [a{0.6h};b ;d]
    job2 = addjob!(pbar; N= Nt)
    @gif for i in 1:Nt÷Nf
        for j in 1:Nf
            ψ = evolution_loop(ψ,potential_x,potential_y,μ,κ1,dt,ϵ)
            σ = hcat(σ, average(ψ,x,dx))
            X = append!(X, x_center(ψ,x,dx))
            Y = append!(Y, y_center(ψ,x,dx))
            update!(job2)
            render(pbar)
        end
        println(sqrt(sum(abs2.(ψ))*dx^2) /sqrt(N_part))
        plt1 = plot(x, x, abs.(ψ),st = :surface,colormap=:vik100,xlabel="y",ylabel="x",clims=(0,3))
        zlims!(0,3)
        xlims!(-dim,dim)
        ylims!(-dim,dim)
        title!("GPE in 2D for $M points, Nt=$Nt and dt=$dt,T=$(round(i*Nf*dt,digits=1))", titlefont=font(10))
        plt2 = plot(t[1:i*Nf],σ[1,3:end],title="σ",label="σx",xlabel="time",xlims=(0,t[end]))
        plot!(t[1:i*Nf],σ[2,3:end],title="σ",label="σy",xlabel="time",xlims=(0,t[end]))
        plot!(t[1:i*Nf],σ[3,3:end],title="σ",label="σ",xlabel="time",xlims=(0,t[end]))
        plt3 = plot(t[1:i*Nf],[X[2:end] Y[2:end]],title="position",label=["x center" "y center"],xlabel="time",ylims=(-3,3),xlims=(0,t[end]))
        
        plot(plt1,plt2,plt3,layout=l,size=(1200,800))
    end
    maxima_index = [1]
    maxima_index = append!(maxima_index, findall(j -> σ[j] < σ[j-1] && σ[j] < σ[j+1], 2:length(σ)-1))
    pulsation = 0
    for i in 1:length(maxima_index)-1
        pulsation = 2π/(t[maxima_index[i+1]] - t[maxima_index[i]])
        println("temps minimum = $(t[maxima_index[i]])")
    end
end

main()
