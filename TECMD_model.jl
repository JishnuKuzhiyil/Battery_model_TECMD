    """
            TECMD model implimentation in Julia.
            Classical first order ECM model with thermal dynamics is assisted with diffusion dynamics.
            The model is also capable of simulating voltage hold phase.
            formulated in Xiong-2022 paer in Applied Energy Journal.

            Implimented by Jishnu Ayyangatu Kuzhiyil
            PhD @ WMG, University of Warwick
            May 2023

    """

    using Plots,DifferentialEquations,SparseArrays

"""_________________________________________________ User inputs_________________________________________________"""


    # Experiment

    Exp_step_no = [1]
    N_cycles = 3
    Experiment = [(false,x->1.6,"Voltage", 2.5,5*3600.0,nothing);repeat([(false,x->0.0,"Time", 3600,5000,nothing),(false,x->-1.6,"Voltage", 4.2,3600*10,nothing),(true,x->-1.6,"Abs_Current", 0.25,50000,4.2),(false,x->0.0,"Time", 3600,50000,nothing),(false,x->5.0,"Voltage", 2.5,3600*1.3,"RPT-Capacity")],N_cycles)]
    Exp_step_transition = [0.0,0.0,0.0,0.0]
    T_events = []
    
    
    # Model parameters

    R₀_ref = 20.611 *1e-3  #Ω
    R_ref = 12.5041 *1e-3  #Ω
    τ_ref = 0.189          #s
    θ_ref = 41.6          #100s

    Ea_R₀ = 8359.94 #J/mol
    Ea_R =  9525.26 #J/mol
    Ea_θ = 25000.0  #J/mol

    Q = 4.84*3600  #As
    T_ref = 298.15 #K
    C₁ = -0.0015397895     
    C₂ = 0.020306583 

    T_ini = 298.15 #K
    T∞ = 298.15 #K


    # Initial conditions and parameetrs vector

    p = [Exp_step_no,Experiment,Exp_step_transition, R₀_ref, R_ref, τ_ref, θ_ref, Ea_R₀, Ea_R, Ea_θ, Q, T_ref, C₁, C₂,T∞]
    u0 = [100.0 ,0.0,0.0,0.0,0.0,T_ini,0.0] #SOC, ΔSOC, q1, q2, V, T,I_hold
    #Solver settings
    dt_max = 10.0
    tspan = (0.0,3600.0*5.0)   # Make this step longer than the first step in the experiment



"""_________________________________________________ Model_________________________________________________"""


    @inline function OCV_LGM50_full_cell(SOC) # SOC ∈ [0,100] #OCP function
            
        OCV =@.(3.132620508717038 + 0.03254176983890392*SOC - 0.003901708378797115*SOC^2 + 
        0.001382745468407752*SOC^3 - 0.00026033289859565573*SOC^4 + 2.7051704798205416e-5*SOC^5 -
        1.753670407892406e-6*SOC^6 + 7.619006342039215e-8*SOC^7 - 2.3099431268369483e-9*SOC^8 +
        4.990071985886017e-11*SOC^9 - 7.728673298552951e-13*SOC^10 + 8.517228717429399e-15*SOC^11 - 
        6.51737840308998e-17*SOC^12 + 3.2902385347157566e-19*SOC^13 - 9.851142596586927e-22*SOC^14 +
        1.3245328408180436e-24*SOC^15) #Broadcasted for faster calculation

    end 

    @inline Arrhenius(T, Ea, Pre_exponent,  T_ref = 298.15) = (Pre_exponent * exp((Ea/8.314) * (1/T - 1/T_ref)))

    function TECMD_model!(du,u,p,t)  #Although allocating form is used, it is stack allocation (StaticArrays.jl), hence faster.

        
        step=p[1][1]
        t_last=p[3][1]
        I = p[2][step][2](t-t_last) .-u[7]
        
        R₀_ref, R_ref, τ, θ_ref, Ea_R₀, Ea_R, Ea_θ, Q, T_ref, C₁, C₂,T∞ = p[4:end]
        SOC, ΔSOC, q1, q2, V, T, I_hold = u

        R = Arrhenius(T, Ea_R, R_ref, T_ref)
        R₀ = Arrhenius(T, Ea_R₀, R₀_ref, T_ref)
        θ = Arrhenius(T, Ea_θ, θ_ref, T_ref)
        
        #SOCdynamics______________________________________________________________
        du[1] = -(100.0/Q)*I       # Coulomb counting, SOC ∈ [0,100]

        #Diffusion dynamics_______________________________________________________
        du[2] = (-5.94/θ)     * ΔSOC  + q1 - (1100.0/Q)    *  I
        du[3]   = (-4.5045/θ^2) * ΔSOC  + q2 - (1716.0/θ/Q)  *  I
        du[4]   = (-0.6757/θ^3) * ΔSOC       - (450.5/θ^2/Q) *  I


        #Overpotential____________________________________________________________
        du[5] = (-1.0/τ) * V - (R/τ) * I # RC branch

        #Temperature dynamics_____________________________________________________
        du[6] = C₁ * (T - T∞) - C₂ * (V - I *R₀) * I# Newton's law of cooling

        #Hold current_____________________________________________________________

        if p[2][step][1] == false
            du[7] = u[7]

        else
             Voltage = OCV_LGM50_full_cell(SOC.+ΔSOC) .+ V .- I * R₀
             du[7]= p[2][step][6] .- Voltage
        end 


    
    end 



    function condition(u, t, integrator)

        step=integrator.p[1][1]
        t_last=integrator.p[3][1]
        I = integrator.p[2][step][2](t-t_last) .- u[end]

        experiment=integrator.p[2][:]
        exp_step=experiment[step]


        if exp_step[1]

            if exp_step[3]=="Time"

                return (t-t_last)-exp_step[4]

            elseif exp_step[3]=="Abs_Current"
                
                return abs(integrator.p[2][step][2](t-t_last) .- u[end]) .- exp_step[4]

            end

        else

            if exp_step[3]=="Time"

                return (t-t_last)-exp_step[4]

            elseif exp_step[3]=="Voltage"
                
                R₀ = integrator.p[4] * exp((integrator.p[8]/8.314) * (1.0 ./u[6] .- 1.0 ./integrator.p[12]))
                return OCV_LGM50_full_cell(u[1]+u[2]) .+ u[5] .- I .* R₀ - exp_step[4]

            end

        end             

    end 

    function affect!(integrator)

        push!(T_events, integrator.t)
        t_secondlast=integrator.p[3][1]
        t_last=integrator.t  
        step_nu=integrator.p[1][1]
        I_last=integrator.p[2][step_nu][2](t_last - t_secondlast)-integrator.u[end]
        

        if step_nu == length(integrator.p[2])
    
            #Stop when the last step in the experiment is completed
            println("the solution has stopped")
            terminate!(integrator)

        else
        
        
            if integrator.p[2][step_nu][1]

                integrator.u[7]=0.0
                nxt_stp_duration=integrator.p[2][step_nu+1][5]
                integrator.p[1][1]=step_nu.+1
                integrator.p[3].=[t_last,t_secondlast,0.0,I_last]
                reinit!(integrator,integrator.u,t0=t_last,tf=t_last+nxt_stp_duration,erase_sol=false)
    
            elseif integrator.p[2][step_nu+1][1]
              
                nxt_stp_duration=integrator.p[2][step_nu+1][5]
                integrator.p[3].=[t_last,t_secondlast,0.0,I_last]
                integrator.p[1][1]=step_nu.+1
                #integrator.u[end]=I_last
            
                reinit!(integrator,integrator.u,t0=t_last,tf=t_last+nxt_stp_duration,erase_sol=false)
    
            else
               
                integrator.p[1][1]=step_nu.+1
                integrator.p[3].=[t_last,t_secondlast,0.0,I_last]
                nxt_stp_duration=integrator.p[2][step_nu+1][5]

                #integrator.opts.dtmax=300.0

                reinit!(integrator,integrator.u,t0=t_last,tf=t_last+nxt_stp_duration,erase_sol=false)
            end
        end
        
    end

    cb=ContinuousCallback(condition,affect!)

"""____________________________________________________Solve the model____________________________________________"""



    Inertia_matrix=spdiagm([ones(6);1.0])
    DAE_system=ODEFunction(TECMD_model!,mass_matrix=Inertia_matrix)
    prob=ODEProblem(DAE_system,u0,tspan,p)
    using BenchmarkTools
    sol = solve(prob,Rosenbrock23(),callback=cb,dtmax=dt_max) 



   


"""__________________________________________________Voltage Calculation___________________________________________"""


    function Current_vals(t_vector,experiment,T_events=T_events)

        step = 1
        t_last_step = 0.0
        I_vector = similar(t_vector)
        
        for i in eachindex(t_vector[1:end])

            if t_vector[i]<=T_events[step]
            I_vector[i] = experiment[step][2](t_vector[i]-t_last_step)
            
            else 
                step +=1
                t_last_step = T_events[step-1]
                I_vector[i] = experiment[step][2](t_vector[i]-t_last_step)
            end 

        end 

        return I_vector
    end 
                
    function Voltage_calculation(solln,I_vector; R₀_ref = R₀_ref, Ea_R₀ = Ea_R₀ , T_ref = T_ref)

        OCV = OCV_LGM50_full_cell.(@view(solln[1,:]) .+  @view(solln[2,:]) )
        V_over = @view(solln[5,:]) - I_vector .* R₀_ref .* exp.((Ea_R₀/8.314) * (1.0 ./@view(solln[6,:]) .- 1.0 ./T_ref))
        V_terminal = OCV .+ V_over

        return V_terminal, OCV, V_over

    end 

    I_vector = Current_vals(sol.t,Experiment)
    V_terminal, OCV, V_over = Voltage_calculation(Array(sol),I_vector)

    function plot_results(t,V_terminal, OCV, I_vector, Temp_vector)

        # Create a plot with three subfigures
        p = plot(layout=(2,2), size=(800,600))

        # Plot V_terminal vs. t in the first subfigure
        plot!(p[1], t./3600.0, V_terminal, xlabel="Time (hrs)", ylabel="Terminal Voltage (V)",legend = false)

        # Plot OCV vs. t in the second subfigure
        plot!(p[2], t./3600.0, OCV, xlabel="Time (hrs)", ylabel="OCV (V)",legend = false)

        plot!(p[3], t./3600.0, I_vector, xlabel="Time (hrs)", ylabel="Current (A)",legend = false)

        plot!(p[4], t./3600.0, Temp_vector .-273.15, xlabel="Time (hrs)", ylabel="Temperature (ᵒC)",legend = false)


        return p
    end

    plot_results(sol.t,V_terminal, OCV, I_vector, Array(sol)[6,:])