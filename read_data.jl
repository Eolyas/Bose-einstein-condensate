using FFTW, LinearAlgebra, DelimitedFiles, Plots, JLD2, CodecZlib

function main_sigma(filename,nb)
    jldopen(filename, "r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        σx= []
        σy= []
        σ = []
        push!(σx,file["sigmax_1"])
        push!(σy,file["sigmay_1"])
        push!(σ,file["sigma_1"])
        #last_line = file["last_line"]
        for i in 2:Nt#last_line
            if mod(i,round(Int,1÷(dt*100))) == 0
                push!(σx,file["sigmax_$i"])
                push!(σy,file["sigmay_$i"])
                push!(σ,file["sigma_$i"])
            end
        end
        t = LinRange(0,Nt*dt,length(σ))./2pi
        maxima_index = findall(j -> σ[j] < σ[j-1] && σ[j] < σ[j+1], 2:length(σ)-1)
        pulsation = 0
        if length(maxima_index) > 1
            pulsation = 1/(t[maxima_index[2]]-t[maxima_index[1]])
        end

        plot(t,σx,label="σx",xlabel="time /2π",title="pulsation = $(trunc(pulsation,digits=4))")
        plot!(t,σy,label="σy")
        plot!(t,σ,label="σ")
        savefig(nb * "_sigma.png")
    end
end

function main_psi(filename,nb)
    jldopen(filename, "r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        L = file["L"]
        M = file["M"]
        last_line = file["last_line"]
        x = LinRange(-L÷2, L÷2, M)
        index = round(Int,1÷(dt*100))
        t = Vector{Float64}(undef,0)
        push!(t,0)
        ψ = Matrix{ComplexF64}(undef,M,1)
        ψ[:,1] = file["psi_1"][:,M÷2]
        for i in 2:last_line
            if mod(i,index) == 0
                ψ = hcat(ψ,file["psi_$i"][:,M÷2])
                push!(t,dt*i)
            end
        end
        t ./= 2pi
        l = @layout [a;b]
        plt1 = heatmap(t,x,abs2.(ψ),c =:jet1,ylabel="x",xlabel="time /2π",title="density at y=0")
        plt2 = plot(x,abs2.(ψ[:,1]),xlabel="x",ylabel="density",label="t=0")
        temp = 0.5
        i = 1
        while findfirst(>(temp),t) !== nothing && i<6
            plot!(x,abs2.(ψ[:,findfirst(>(temp),t)]),label="t=$(temp)")
            temp += 0.5
            i += 1
        end
        plot(plt1,plt2,layout=l)
        savefig(nb * ".png")
    end
end

function main_gif(filename,nb)
    jldopen(filename, "r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        L = file["L"]
        M = file["M"]
        last_line = file["last_line"]
        index = round(Int,1÷(dt*100))
        x = LinRange(-L÷2, L÷2, M)
        maxi = 0
        for i in 2:last_line
            if mod(i,index) == 0
                ψ = file["psi_$i"]
                temp = maximum(abs2.(ψ))
                if temp > maxi
                    maxi = temp
                end
            end
        end
        anim = @animate for j in round.(Int,range(1,Nt÷index,400))
            i = j*index
            if haskey(file,"psi_$i")
                ψ = file["psi_$i"]
                plot(x,x,abs2.(ψ),st=:surface,c=:jet1,xlabel='x',ylabel='y',title="cylindric evolution at t=$(round(Int,i/2pi)÷dt)",zlims=(0,round(Int,maxi)),clims=(0,round(Int,maxi)))
            end
        end
        gif(anim,nb * "_gif.gif",fps=20)
    end
end

function main_top_view(filename,nb)
    jldopen(filename,"r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        L = file["L"]
        M = file["M"]
        last_line = file["last_line"]
        index = round(Int,1÷(dt*100))
        x = LinRange(-L÷2,L÷2,M)
        maxi = 0
        for i in 2:last_line
            if mod(i,index) == 0
                ψ = file["psi_$i"]
                temp = maximum(abs2.(ψ))
                if temp > maxi
                    maxi = temp
                end
            end
        end
        anim = @animate for j in round.(Int,range(1,Nt÷index,400))
            i = j*index
            ψ = file["psi_$i"]
            heatmap(x,x,abs2.(ψ),c=:jet1,xlabel='x',ylabel='y',title="cylindric evolution at t=$(round(Int,i/2pi)÷dt)",clims=(0,round(Int,maxi)))
        end
        gif(anim,nb * "_gif_top_view.gif",fps=20)
    end
end

function repetition(filename,nb)
    jldopen(filename,"r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        L = file["L"]
        M = file["M"]
        dx = L/M
        N_part = file["N_part"]
        last_line = file["last_line"]
        t = Vector{Float64}(undef,0)
        index = round(Int,1÷(dt*100))
        push!(t,dt)
        x = LinRange(-L÷2,L÷2,M)
        initial = file["psi_1"]
        scalar = []
        coef = dx^2/N_part^2/256
        push!(scalar,abs2(dot(initial,file["psi_1"]))*coef)
        for i in 2:last_line
            if mod(i,index) == 0
                ψ = file["psi_$i"]
                push!(scalar,abs2(dot(initial,ψ))*coef)
                push!(t,dt*i)
            end
        end
        plot(t./2pi,scalar,xlabel="time /2π",ylabel="Fidelity",title="Fidelity during evolution",label="|⟨ψ(0)|ψ(t)⟩|²",legend=:outertopright)
        savefig(nb*"_scalar.png")
    end
end

function ground_state(filename,nb)
    jldopen(filename,"r") do file
        psi = file["psi_1"]
        L = file["L"]
        M = file["M"]
        x = LinRange(-L÷2,L÷2,M)
        N = file["N_part"]
        ω = 1
        g = 2
        μ = sqrt(N*g*ω^2/pi)
        r = x.^2 .+ x'.^2
        n = (μ .- 0.5 * ω^2 * r) ./ g
        nx = @. max(n, 0.0)
        plt1 = heatmap(x,x,abs2.(psi),c=:jet1,xlabel="x",ylabel="y",title="density of the ground state")
        plt2 = heatmap(x,x,nx,c=:jet1,xlabel="x",ylabel="y",title="Thomas-Fermi approximation in 2D")
        sub = abs2.(psi) .- nx
        plt3 = heatmap(x,x,sub,c=:jet1,xlabel="x",ylabel="y",title="Difference between Thomas-Fermi and the ground state")
        plot(plt1,plt2,plt3)
        savefig(nb*"_ground_state.png")
    end
end

function density_plot(filename,nb)
    jldopen(filename, "r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        L = file["L"]
        M = file["M"]
        last_line = file["last_line"]
        x = LinRange(-L÷2, L÷2, M)
        index = round(Int,1÷(dt*100))
        t = Vector{Float64}(undef,0)
        push!(t,0)
        counter = [0]
        for i in 2:last_line
            if mod(i,index) == 0
                push!(t,dt*i)
                append!(counter,i)
            end
        end
        t ./= 2pi
        l = @layout [a b c d]
        plt1 = heatmap(x,x,abs2.(file["psi_1"]),c=:jet1,ylabel="y",xlabel="x",title="t/T=0",clims=(0,40),xlims=(-10,10),ylims=(-10,10),colorbar=:none)
        plt2 = heatmap(x,x,abs2.(file["psi_$(counter[findfirst(>(2/7),t)])"]),c=:jet1,ylabel="y",xlabel="x",title="t/T=2/7",clims=(0,40),xlims=(-10,10),ylims=(-10,10),colorbar=:none)
        plt3 = heatmap(x,x,abs2.(file["psi_$(counter[findfirst(>(1),t)])"]),c=:jet1,ylabel="y",xlabel="x",title="t/T=1",clims=(0,40),xlims=(-10,10),ylims=(-10,10),colorbar=:none)
        plt4 = heatmap(x,x,abs2.(file["psi_$(counter[findfirst(>(2),t)])"]),c=:jet1,ylabel="y",xlabel="x",title="t/T=2",clims=(0,40),xlims=(-10,10),ylims=(-10,10),colorbar=:none)
        # plt1 = heatmap(t,x,abs2.(ψ),c =:jet1,ylabel="x",xlabel="time /2π",title="density at y=0")
        # plt2 = plot(x,abs2.(ψ[:,1]),xlabel="x",ylabel="density",label="t=0")
        # temp = 0.5
        # i = 1
        # while findfirst(>(temp),t) !== nothing && i<6
        #     plot!(x,abs2.(ψ[:,findfirst(>(temp),t)]),label="t=$(temp)")
        #     temp += 0.5
        #     i += 1
        # end
        plot(plt1,plt2,plt3,plt4,layout=l,size=(1600,400))
        savefig(nb * "density_plot.png")
    end
end

function main(filename,nb)
    # main_sigma(filename,nb)
    # main_gif(filename,nb)
    # main_psi(filename,nb)
    # main_top_view(filename,nb)
    # repetition(filename,nb)
    # ground_state(filename,nb)
    density_plot(filename,nb)
end

main("data_cyl_N500_g2.jld2","cyl_500")