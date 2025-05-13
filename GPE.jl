using FFTW, LinearAlgebra, DelimitedFiles, Plots, JLD2, CodecZlib, Dates
gr(show=false)

# Nonlinear step in 2D
function nonlinear_step(ψ,potential,g,dt,ϵ)
    return exp.(-1im .* (g .* abs2.(ψ) .+ potential) .* dt ./ 2 ./ ϵ)
end
function imaginary_nonlinear_step(ψ,potential,g,dt,ϵ)
    return exp.(-(g .* abs2.(ψ) .+ potential) .* dt ./ 2 ./ ϵ)
end

function evolution_loop(ψ,potential,μ,g,dt,ϵ)
    ψ .*= nonlinear_step(ψ,potential,g,dt,ϵ)
    ψf = fft(ψ)
    ψf .*= exp.(-1im*ϵ/2*(μ.^2 .+ μ'.^2)*dt)
    ψff = ifft(ψf)
    ψff .*= nonlinear_step(ψff,potential,g,dt,ϵ)
    return ψff
end
function imaginary_evolution_loop(ψ,potential,μ,g,dt,ϵ)
    ψ .*= imaginary_nonlinear_step(ψ,potential,g,dt,ϵ)
    ψf = fft(ψ)
    ψf .*= exp.(-ϵ ./2 .*(μ.^2 .+ μ'.^2) .*dt)
    ψff = ifft(ψf)
    ψff .*= imaginary_nonlinear_step(ψff,potential,g,dt,ϵ)
    return ψff
end

function average(ψ,x,dx)
    prob_density = abs2.(ψ)
    x_avg = sum(x.*prob_density)*dx^2
    y_avg = sum(x'.*prob_density)*dx^2
    x2_avg = sum(x.^2 .*prob_density)*dx^2
    y2_avg = sum(x'.^2 .*prob_density)*dx^2
    σx = sqrt(abs(x2_avg - x_avg^2))
    σy = sqrt(abs(y2_avg - y_avg^2))
    σ = sqrt(σx^2 + σy^2)
    return σx, σy, σ
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

function box_potential(x,L,V0,steep)
    return V0*(1 ./(1 .+exp.(-(x.-L)/steep)) .+ 1 ./(1 .+exp.((x .+L)/steep)))
end

function cylindrical_potential(x,R,V0,steep)
    r = sqrt.(x.^2 .+x'.^2)
    return -V0 ./(1 .+exp.((r .- R) ./ steep)) .+V0
end

function triangle_potential(x,R,V0,steep)
    
end

function nx(x,N,g,ω)
    μ = sqrt(N*g*ω^2/pi)
    r = x.^2 .+ x'.^2
    n = (μ .- 0.5 * ω^2 * r) ./ g
    return @. max(n, 0.0)
end

function test_same(g,N_part,dt,L,M)
    jldopen("data_cylindric.jld2","a+") do file
    end
    jldopen("data_cylindric.jld2","r") do file
        if haskey(file,"Nt")
            println("has Nt")
            if file["g"] == g && file["N_part"] == N_part && file["dt"] == dt && file["L"] == L && file["M"] == M
                println("true inside function")
                return true
            else
                return false
            end
        else
            return false
        end
    end
end

# Main function to run the simulation
function main()
    # Initial parameters
    ϵ = 1.0
    ω = 1.0
    ω_var = 1.0
    x0 = 0
    g = 2.0
    N_part = 400

    h = 1/16
    a = -14.0 # Spatial minimum for both x and y
    b = -a # Spatial maximum for both x and y
    L = b - a
    M = Int(L/h)
    dx = (b-a)/M
    dt = 0.00001  # Time step
    Nt = 1200000  # Number of time steps
    t = LinRange(0,Nt*dt,Nt)
    x = LinRange(a, b, M)

    V0 = 10e10
    steep = 0.04
    pot_size = 4

    potential = ω^2 * x.^2/2
    μ = 2 * π / L * vcat(0:(M/2-1),-M/2:-1)

    same = test_same(g,N_part,dt,L,M)
    println(same)
    if !same
        # Initialize wavefunction ψ
        ψ = exp.(-x.^2 .- x'.^2)
        #ψ .= 1/sqrt(π*ϵ) * exp.((-x.^2 .- x'.^2)/(2*ϵ))  # Initial condition
        #ψ .= γ^(1/4)/sqrt(π*ϵ) * exp.((-x.^2 .- γ .* x'.^2)/(2*ϵ)) * exp.(1im .*cosh.(sqrt.(x .^2 .+ 2 .* x'.^2)) ./ϵ)
        ψ ./= sqrt(sum(abs2.(ψ))*dx^2) / sqrt(N_part) # Normalize the initial wavefunction
        for i in 1:Nt÷2
            #ψ = imaginary_evolution_loop(ψ,(potential .+ potential'),μ,g,dt,ϵ)
            ψ = imaginary_evolution_loop(ψ,box_potential(x,pot_size,V0,steep).+box_potential(x,pot_size,V0,steep)',μ,g,dt,ϵ)
            #ψ = imaginary_evolution_loop(ψ,cylindrical_potential(x,pot_size,V0,steep),μ,g,dt,ϵ)
            ψ ./= sqrt(sum(abs2.(ψ))*dx^2) / sqrt(N_part)
        end

        ψ_real = abs2.(ψ)
        plot(x,ψ_real[:,M ÷ 2],xlabel="x",title="fondamental",label="density")
        plot!(x,angle.(ψ[:,M÷2]),label="angle")
        plot!(x,potential,label="harmonic potential")
        savefig("cylindric_fondamental_N=$(N_part)_g=$(g).png")
        # TF = nx(x,N_part,g,ω)[:,M ÷ 2]
        # plot!(x,TF,label="Thomas-Fermi",)
        # plot!(x,angle.(ψ)[:,M ÷ 2],label="angle")
        # if maximum(TF) < maximum(abs2_ψ)
        #     ylims!((0,maximum(abs2_ψ)))
        # else
        #     ylims!((0,maximum(TF)))
        # end
        # savefig("Difference_bwt_theory_&_GPE_dt$(dt)_Nt$(Nt)_L$(L)_h$(h)_g$(g)_N$(N_part).svg")
    end

    potential = ω_var^2 *(x.-x0).^2/2

    if same
        jldopen("data_cylindric.jld2","a+",compress=true) do file
            last_line = file["last_line"]
            ψ = file["psi_$(last_line)"]
            for i in last_line+1:Nt
                ψ = evolution_loop(ψ,(potential .+potential'),μ,g,dt,ϵ)
                if mod(i,round(Int,1÷(dt*100))) == 0
                    σx, σy, σ = average(ψ,x,dx)
                    file["psi_$i"] = ψ
                    file["sigmax_$i"] = σx
                    file["sigmay_$i"] = σy
                    file["sigma_$i"] = σ
                    delete!(file,"last_line")
                    file["last_line"] = i
                    println("written more line for t=$i")
                end 
            end
            println("finished old calc")
        end
    else
        jldopen("data_cylindric.jld2","w",compress=true) do file
            file["g"] = g
            file["N_part"] = N_part
            file["Nt"] = Nt
            file["dt"] = dt
            file["L"] = L
            file["M"] = M
            σx, σy, σ = average(ψ,x,dx)
            file["sigmax_1"] = σx
            file["sigmay_1"] = σy
            file["sigma_1"] = σ
            file["psi_1"] = ψ
            file["last_line"] = 1
            for i in 2:Nt
                ψ = evolution_loop(ψ,(potential .+potential'),μ,g,dt,ϵ)
                if mod(i,round(Int,1÷(dt*100))) == 0
                    σx, σy, σ = average(ψ,x,dx)
                    file["psi_$i"] = ψ
                    file["sigmax_$i"] = σx
                    file["sigmay_$i"] = σy
                    file["sigma_$i"] = σ
                    delete!(file,"last_line")
                    file["last_line"] = i
                    println("written new line for t=$i")
                end 
            end
            println("finished new calc")
        end
    end
end

main()
println(now())