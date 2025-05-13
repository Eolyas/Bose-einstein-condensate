using FFTW, LinearAlgebra, DelimitedFiles, Plots, JLD2, CodecZlib

function main_sigma()
    jldopen("data_cylindric.jld2", "r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        σx= []
        σy= []
        σ = []
        push!(σx,file["sigmax_1"])
        push!(σy,file["sigmay_1"])
        push!(σ,file["sigma_1"])
        for i in 2:Nt
            if mod(i,round(Int,1÷(dt*100))) == 0
                push!(σx,file["sigmax_$i"])
                push!(σy,file["sigmay_$i"])
                push!(σ,file["sigma_$i"])
            end
        end
        t = LinRange(0,Nt*dt,length(σ))
        maxima_index = findall(j -> σ[j] < σ[j-1] && σ[j] < σ[j+1], 2:length(σ)-1)
        pulsation = 0
        if length(maxima_index) >= 1
            pulsation = 2π/(t[maxima_index[1]])
        end

        plot(t./2pi,σx,label="σx",xlabel="time (2π)",title="pulsation = $(trunc(pulsation,digits=4))")
        plot!(t./2pi,σy,label="σy")
        plot!(t./2pi,σ,label="σ")
        savefig("sigma_cylindric.png")
    end
end

function main_psi()
    jldopen("data_cylindric.jld2", "r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        L = file["L"]
        M = file["M"]
        x = LinRange(-L÷2, L÷2, M)
        index = round(Int,1÷(dt*100))
        t = Vector{Float64}(undef,0)
        push!(t,0)
        ψ = Matrix{ComplexF64}(undef,M,1)
        ψ[:,1] = file["psi_1"][:,M÷2]
        for i in 2:Nt
            if mod(i,index) == 0
                ψ = hcat(ψ,file["psi_$i"][:,M÷2])
                push!(t,dt*i)
            end
        end
        l = @layout [a;b]
        plt1 = heatmap(t,x,abs2.(ψ),c =:jet1,ylabel="x",xlabel="time (2π)",title="density at y=0")
        plt2 = plot(x,abs2.(ψ[:,1]),xlabel="x",ylabel="density",label="t=0")
        temp = 0.5
        i = 1
        while findfirst(>(temp),t) !== nothing && i<6
            plot!(x,abs2.(ψ[:,findfirst(>(temp),t)]),label="t=$(temp)")
            temp += 0.5
            i += 1
        end
        plot(plt1,plt2,layout=l)
        savefig("cylindric.png")
    end
end

function main_gif()
    jldopen("data_cylindric.jld2", "r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        L = file["L"]
        M = file["M"]
        index = round(Int,1÷(dt*100))
        x = LinRange(-L÷2, L÷2, M)
        maxi = 0
        for i in 2:Nt
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
            plot(x,x,abs2.(ψ),st=:surface,c=:jet1,xlabel='x',ylabel='y',title="cylindric evolution at t=$(round(Int,i/2pi))",zlims=(0,round(Int,maxi)),clims=(0,round(Int,maxi)))
        end
        gif(anim,"gif_cylindric.gif",fps=20)
    end
end

function main_top_view()
    jldopen("data_cylindric.jld2","r") do file
        Nt = file["Nt"]
        dt = file["dt"]
        L = file["L"]
        M = file["M"]
        index = round(Int,1÷(dt*100))
        x = LinRange(-L÷2,L÷2,M)
        maxi = 0
        for i in 2:Nt
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
            heatmap(x,x,abs2.(ψ),c=:jet1,xlabel='x',ylabel='y',title="cylindric evolution at t=$(round(Int,i/2pi))",clims=(0,round(Int,maxi)))
        end
        gif(anim,"gif_cylindric_top_view.gif",fps=20)
    end
end

function main()
    main_sigma()
    main_gif()
    main_psi()
    main_top_view()
end

main()