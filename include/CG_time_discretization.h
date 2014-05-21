//
//  explicit_time_discretization.templates.h
//  
//
//  Created by Scott Miller on 10/24/12.
//  Copyright 2012 Applied Research Lab, Penn State. All rights reserved.
//

using namespace dealii;

typedef dealii::Vector<double> ParVec;


template <int dim>
class TimeDiscretization : public dealii::Subscriptor
{
public:
    enum DiscretizationType {
        BackwardEuler,
        Bathe,
        CrankNicolson,
        RK1,
        RK2,
        RK3,
        RK4,
        RK5,
        BDF2,
        SSP_5_4};
    
private:
    
    bool implicit; // implicit? or explicit
    
        //! Current timestep size
    double delta_t;
    
        //! Number of timesteps 
    unsigned int n_timestep;
    
        //! Number of RK stages
    unsigned int n_stages;
    
        //! Set the current time
    double cur_time;
    
        //! Final time of the simulation
    double final_time;
    
        //! Time scheme to use.
        //! Integer refers to enumeration list
    DiscretizationType time_scheme;
    
    
        //! Store a few previous time step sizes, depending
        //! on the time discretization we use.
        //! Necessary for non-constant time step sizes.
    std::vector<double> prev_delta_t;
    
        //! Is the mesh topology changing?
        //! Currently this class only deals well when the value == false
        //! If mesh_changed==true, we shall throw an error of ExcNotImplemented
    bool mesh_changed;
    
        //! Number of dofs.  Constant in the current incarnation, as mesh_changed=false
    unsigned int n_dofs;
    
        //! Solution vectors, current through N-previous.
    //! N=1 for single step or RK methods
    //! N>1 for linear multistep methods
    std::vector<ParVec>         solution;
    
        //! Vectors for stages in RK methods
    std::vector<ParVec>        stages;
    
        //! For SSP methods I need more vectors!
    std::vector<ParVec>        u_i;
    
        //! Number of previous time steps to keep
    unsigned int num_old_times;
    
        //! We store some functionals to compute the RHS and check convergence:
        //! We set these in the constructor.
        //! These functions are provided by the `spatial discretization'
    dealii::std_cxx1x::function<void (ParVec&, ParVec&,
                                    bool, double)>
                        assemble_rhs;
                        
    dealii::std_cxx1x::function<void (ParVec&, ParVec&,
                                    bool, double)>
                        assemble_Bathe;
    
    dealii::std_cxx1x::function<void (ParVec&, ParVec&)> 
                        assemble_mass_matrix_and_multiply;
    
    dealii::std_cxx1x::function<bool (unsigned int, ParVec&)> 
                        check_convergence;
    
    dealii::std_cxx1x::function<void (int)>            
                        init_newton_iteration;
    
    dealii::std_cxx1x::function<void (ParVec&)> 
                        applyLimiters;
    
    
        //! Internal pointers
    ParVec* current_solution;
    ParVec* old_solution;
    
    ParVec* solution_derivative;
    
    DoFHandler<dim> &dh;
    
//    dealii::Vector<double> newton_update, residual, mass_residual;
    
        //! advance functions for specific schemes
    void advance_RK1 (void);
    void advance_RK2 (void);
    void advance_RK3 (void);
    void advance_RK4 (void);
    void advance_RK5 (void);
    void advance_SSP_5_4 (void);
    void advance_Bathe (void);
    
    void compute_residual (unsigned int u_i_index, unsigned int stage_index);
    
public:
    
        //! Constructor
    TimeDiscretization (const dealii::std_cxx1x::function<void (ParVec&, ParVec&,
                                                           bool, double)>                    
                                                    &assemble_rhs,
                    	const dealii::std_cxx1x::function<void (ParVec&, ParVec&,
                                                           bool, double)>                    
                                                    &assemble_Bathe,
                        DoFHandler<dim> &dof_handler );
    
        //! Destructor
    ~TimeDiscretization ();
    
        //! Set pointer to the vector space:
//    inline void set_vector_space(VectorSpace<dim> &vs){vspace = &vs;};
    
    /** Reinit. */
    void reinit(unsigned int n_dofs, double time=-1.); 
    
    //! Initialize functions; set up necessary data structures
    void initialize (void);
    
        //! Query timestep size
    inline double time_step_size (void){return delta_t;};
    
        //! Query timestep number
    inline unsigned int time_step_num (void){return n_timestep;};
    
        //! Access current time
    inline double current_time (void){return cur_time;};
    
		//! Select time integration method
	void set_time_scheme (std::string scheme);
	
	void set_delta_t(double t){delta_t = t;};
	void set_final_time(double t){final_time = t;}
    
        //! Set flag for changing topology
    inline void set_topology_changed (bool flag) {mesh_changed = flag;};
    
        //! Access solution vector used to compute the spatial discretization
        //! This would be the t^{n+1} solution for fully implicit, 
        //! or the t^n solution for explicit (although for more general methods,
        //! we will need multiple "old" solutions... TODO:  add this capability
    ParVec& access_current_solution (void){return *current_solution;};
    ParVec& access_solution_derivative (void){return *solution_derivative;};
    
    const ParVec& access_current_solution (void) const {return *current_solution;};
    
        //! Soley for setting initial conditions
    ParVec& access_old_solution (void){return *old_solution;};
    
    void set_current_from_old (void){*(this->current_solution) = *(this->old_solution);};
    void set_old_from_current (void){*(this->old_solution) = *(this->current_solution);};
    
        //! Are we at the final time yet?
    inline bool finalized (void) 
    {
        if (cur_time < final_time)
            return false;
        
        return true;
    }//finalized
    
        //! Advance one time step
    void advance (void);
    
};//class-TimeDiscretization

/** Full constructor */
template <int dim>
TimeDiscretization<dim>::TimeDiscretization (
//         MPI_Comm& mpi_communicator,
         const dealii::std_cxx1x::function<void (ParVec&, ParVec&, bool, double)>                    
                                        &assemble_rhs,
    	const dealii::std_cxx1x::function<void (ParVec&, ParVec&, bool, double)>                    
                                        &assemble_Bathe,
                        DoFHandler<dim> &dof_handler)
        
: n_timestep(0),
  cur_time(-1.0),
  dh(dof_handler)
//       Utilities::System::get_this_mpi_process(MPI_COMM_WORLD)==0)
{
    this->assemble_rhs = assemble_rhs;
    this->assemble_Bathe = assemble_Bathe;
   //  this->assemble_mass_matrix_and_multiply = assemble_mass_matrix_and_multiply;
//     this->check_convergence = check_convergence;
//     this->init_newton_iteration = init_newton_iteration;
//     this->applyLimiters = applyLimiters;
}//constructor

/** Destructor. */
template <int dim>
TimeDiscretization<dim>::~TimeDiscretization ()
{}

/** Reinit. */
template <int dim>
void TimeDiscretization<dim>::reinit(unsigned int dofs, double time)
{
    n_dofs = dofs;
    
    if (time < -0.50)
        initialize();
    else
    {
        cur_time = time;
            // Reinitialize data for new # of dofs, but don't clear the solution vector!
        for(unsigned int i=0; i<stages.size(); ++i)
            stages[i].reinit(n_dofs);
    }
    
    initialize();
    
}//reinit

/** set_time_scheme. */
template <int dim>
void TimeDiscretization<dim>::set_time_scheme(std::string scheme)
{
    if (scheme == "RK1")
        {
            time_scheme = RK1;
            n_stages = 1;
        }  
    else if (scheme == "RK2")
        {
            time_scheme = RK2;
            n_stages = 2;
        }
    else if (scheme == "RK3")
        {
            time_scheme = RK3;
            n_stages = 3;
        }
    else if (scheme == "RK4")
        {
            time_scheme = RK4;
            n_stages = 4;
        }
    else if (scheme == "RK5")
        {
            time_scheme = RK5;
            n_stages = 5;
        }
    else if (scheme == "SSP_5_4")
        {
            time_scheme = SSP_5_4;
            n_stages = 5;
        }
    else if (scheme == "Bathe")
    	{
    		time_scheme = Bathe;
    		n_stages = 2;
    	}
    else
        {
        std::cout<< sc::warning << sc::nl 
            << "TIME SCHEME = "<< scheme 
            << " NOT FOUND" << sc::nl;
        
        Assert(false, ExcNotImplemented());           
        }

}//set_time_scheme

/** initialize()
 *  - size vectors using info obtained
 *    in set_time_scheme, etc
 */
template <int dim>
void TimeDiscretization<dim>::initialize ()
{
    std::cout<<"\nInitializing "<<n_stages<<"-stage Explicit Time Integrator with n_dofs = "<<n_dofs<<std::endl;
        
        // this class is all explicit schemes
    implicit = false;
    
        //! Need 2 time levels/solution vectors
        //! Hence, only 1 old time
    num_old_times = 1;
    
//    ParVec vec(vspace->locally_owned_dofs,
//               vspace->locally_relevant_dofs,
//               vspace->mpi_communicator);
    
    ParVec vec(n_dofs);
    
    vec = 0.0;
    
    solution.push_back(vec);//current_solution
    solution.push_back(vec);//old_solution
    
    current_solution = &(solution[0]);
    old_solution     = &(solution[1]);
    
        //! Vectors for k_i stages
    for(unsigned int ns=0; ns<=n_stages; ++ns)
    {
        stages.push_back(vec);
        u_i.push_back(vec);
    }
    
    if(time_scheme == Bathe){
    	solution_derivative = &(stages[1]);
    }
    
}//initialize

template <int dim>
void TimeDiscretization<dim>::compute_residual (unsigned int u_i_index, unsigned int stage_index)
{	
        //! Make call to SD so it can do what it needs to 
        //! for each newton iteration/explicit stage
    //init_newton_iteration(stage_index);
    
    stages[stage_index] = 0.0;
    
    assemble_rhs(u_i[u_i_index],
                 stages[stage_index],
                 false,
                 1.0);
    
//    std::cout<<"\nNorm of RHS residual = "<<(stages[stage_index]).l2_norm()<<std::endl;
    
//    assemble_mass_matrix_and_multiply(u_i[u_i_index],
//                                      stages[stage_index]);
    
}//end-compute_residual


template <int dim>
void TimeDiscretization<dim>::advance(void)
{
    if (time_scheme == RK1)
        return advance_RK1();
    else if (time_scheme == RK2)
        return advance_RK2();
    else if (time_scheme == RK3)
        return advance_RK3();
    else if (time_scheme == RK4)
        return advance_RK4();
    else if (time_scheme == SSP_5_4)
        return advance_SSP_5_4();
    else if (time_scheme == Bathe)
    	return advance_Bathe();
    else
    {
        std::cout<< sc::warning << sc::nl 
        << "advance() not implemented for time scheme = "<< time_scheme << sc::nl;
        
        Assert(false, ExcNotImplemented());
    }
        
}//advance

/**********************************************
 * Bathe := 2nd order implicit
 *
 **********************************************/
template <int dim>
void TimeDiscretization<dim>::advance_Bathe(void)
{
	n_timestep++;
    
    double old_time = cur_time;
    if(n_timestep % 1000 == 0)
    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
            << "Global time = " << cur_time << sc::nl;
            
    set_old_from_current();
    
    u_i[0] = *(this->old_solution);
    u_i[0] *= 4./delta_t;
    u_i[0] += *(this->solution_derivative);
    
    stages[0] = 0.0;
    
    cur_time = old_time + delta_t*0.5;
    
    assemble_Bathe(u_i[0],
                 stages[0],
                 true,
                 4./delta_t);
                 
    u_i[1] = stages[0];//u^{n+.5}
    u_i[1] *= 4.;
    u_i[1] -= *(this->old_solution);
    u_i[1] /= delta_t;
    
    cur_time = old_time + delta_t;
    
    // save u^{n+0.5} for derivative
    *(this->solution_derivative) = stages[0];
    *(this->solution_derivative) *= (-4.0);
    
    stages[0] = 0.;
    assemble_Bathe(u_i[1],
                 stages[0],
                 true,
                 3./delta_t);
                 
    *(this->current_solution) = stages[0];
    
    stages[0] *= 3.;
	*(this->solution_derivative) += stages[0];
    *(this->solution_derivative) += *(this->old_solution);
    *(this->solution_derivative) /= delta_t;
}


/**********************************************
 * RK1 := Explicit Euler
 *
 *  y^{n+1} = y^{n} + f(t^{n}, y^{n})
 *
 **********************************************/
template <int dim>
void TimeDiscretization<dim>::advance_RK1(void)
{
    n_timestep++;
    
    double old_time = cur_time;
    if(n_timestep % 1000 == 0)
    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
            << "Global time = " << cur_time << sc::nl;
    
//    applyLimiters(*(this->current_solution));
//    current_solution->update_ghost_values();
    
        //! Set the old_solution equation to the current
        //! The "current" solution is then the initial guess
        //! for the Newton iterations
    set_old_from_current();
    
        //! Create temp vector
    ParVec &tmp_vec = stages[1];
    
    u_i[0] = *(this->current_solution);
//    u_i[0].update_ghost_values();
    
        //! Stage 1
//        std::cout<<"\nBefore compute_residual"<<std::endl;
    compute_residual(0,0);
//    stages[0].update_ghost_values();
    tmp_vec = stages[0];
    tmp_vec *= (delta_t);
    u_i[1] = tmp_vec;
    u_i[1] += u_i[0];
    
    cur_time = old_time + delta_t;
    
    ConstraintMatrix constraints;
    
    VectorTools::interpolate_boundary_values(dh, 
										0,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    if(dim==1)
    	VectorTools::interpolate_boundary_values(dh, 
										1,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    constraints.close();
    constraints.distribute(u_i[1]);
    constraints.clear();		

    
        //    stages[0].update_ghost_values();
//    u_i[1].update_ghost_values();
    
//    applyLimiters(u_i[1]);
    
        //! Final solution for the timestep
    *(this->current_solution) = u_i[1];

}//advance_RK1

template <int dim>
void TimeDiscretization<dim>::advance_RK2(void)
{
//    n_timestep++;
//    
//    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
//                        << "Global time = "<< cur_time << sc::nl;
//    
//        //! Set the old_solution equation to the current
//        //! The "current" solution is then the initial guess
//        //! for the Newton iterations
//    set_old_from_current();
//    
//        //! Create temp vector
//    Vector<double> &tmp_vec = stages[2];
//    tmp_vec = *(this->current_solution);
//    
//        //! Stage 1
//    NewtonRaphson_RK(false);
//    stages[0] = *(this->current_solution);
//        // current_solution is really k1, adjust accordingly
//    *(this->current_solution) *= (0.5*delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//        //! Update time for stages 2 and 3
//    cur_time += 0.5*delta_t;
//    
//        //! Stage 2
//    NewtonRaphson_RK(false);
//    stages[1] = *(this->current_solution);
//    
//    cur_time += 0.5*delta_t;
//    
//        //! Set final solution
//    residual = stages[1];
//    residual *= delta_t;
//    set_current_from_old();
//    *(this->current_solution) += residual;
}

template <int dim>
void TimeDiscretization<dim>::advance_RK3(void)
{
//    n_timestep++;
//    
//    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
//    << "Global time = "<< cur_time << sc::nl;
//    
//        //! Set the old_solution equation to the current
//        //! The "current" solution is then the initial guess
//        //! for the Newton iterations
//    set_old_from_current();
//    
//        //! Create temp vector
//    Vector<double> &tmp_vec = stages[3];
//    tmp_vec = *(this->current_solution);
//    
//        //! Stage 1
//    NewtonRaphson_RK(false);
//    stages[0] = *(this->current_solution);
//        // current_solution is really k1, adjust accordingly
//    *(this->current_solution) *= (0.5*delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 1 finished"<<sc::nl;
//    
//        //! Update time for stage 2
//    cur_time += 0.5*delta_t;
//    
//        //! Stage 2
//    NewtonRaphson_RK(false);
//    stages[1] = *(this->current_solution);
//        // current_solution is really k2, adjust accordingly
//    *(this->current_solution) *= (2.0*delta_t);
//    tmp_vec = stages[0];
//    tmp_vec *= -2.0;
//    tmp_vec += *(this->old_solution);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 2 finished"<<sc::nl;
//    
//        //! Stage 3
//        //! Update time for stage 3
//    cur_time += 0.5*delta_t;
//    NewtonRaphson_RK(false);
//    stages[2] = *(this->current_solution);
//    
//    std::cout<<"Stage 3 finished"<<sc::nl;
//    
//        //! Compute the solution at the end of the timestep
//        // abuse my residual vector a little bit:
//    residual = stages[1];
//    residual *= 4.0;
//    residual += stages[0];
//    residual += stages[2];
//    residual *= (delta_t/6.0);
//    set_current_from_old();
//    *(this->current_solution) += residual;
}

template <int dim>
void TimeDiscretization<dim>::advance_RK4(void)
{
//    n_timestep++;
//    
//    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
//    << "Global time = "<< cur_time << sc::nl;
//    
//        //! Set the old_solution equation to the current
//        //! The "current" solution is then the initial guess
//        //! for the Newton iterations
//    set_old_from_current();
//    
//        //! Create temp vector
//    Vector<double> &tmp_vec = stages[4];
//    tmp_vec = *(this->current_solution);
//    
//        //! Stage 1
//    NewtonRaphson_RK(false);
//    stages[0] = *(this->current_solution);
//        // current_solution is really k1, adjust accordingly
//    *(this->current_solution) *= (0.5*delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 1 finished"<<sc::nl;
//    
//        //! Update time for stages 2 and 3
//    cur_time += 0.5*delta_t;
//    
//        //! Stage 2
//    NewtonRaphson_RK(false);
//    stages[1] = *(this->current_solution);
//        // current_solution is really k2, adjust accordingly
//    *(this->current_solution) *= (0.5*delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 2 finished"<<sc::nl;
//    
//        //! Stage 3
//    NewtonRaphson_RK(false);
//    stages[2] = *(this->current_solution);
//        // current_solution is really k3, adjust accordingly
//    *(this->current_solution) *= (delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 3 finished"<<sc::nl;
//    
//        //! Update time for stage4
//    cur_time += 0.5*delta_t;
//    
//        //! Stage 4
//    NewtonRaphson_RK(false);
//    stages[3] = *(this->current_solution);
//    
//    std::cout<<"Stage 4 finished"<<sc::nl;
//    
//        //! Compute the solution at the end of the timestep
//        // abuse my residual vector a little bit:
//    residual = stages[1];
//    residual += stages[2];
//    residual *= 2.0;
//    residual += stages[0];
//    residual += stages[3];
//    residual *= (delta_t/6.0);
//    set_current_from_old();
//    *(this->current_solution) += residual;
//
}//advance_RK4

template <int dim>
void TimeDiscretization<dim>::advance_RK5(void)
{
    n_timestep++;
    
    double old_time = cur_time;
    if(n_timestep % 1000 == 0)
    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
              << "Global time = " << cur_time << sc::nl;
    
    applyLimiters(*(this->current_solution));
    
        //! Set the old_solution equation to the current
        //! The "current" solution is then the initial guess
        //! for the Newton iterations
    set_old_from_current();
    
        //! Create temp vector
    ParVec &tmp_vec = stages[5];
    
    u_i[0] = *(this->current_solution);
    
        //!  Note:  stages[j] = RHS(u_i[j])
    
        //! NON-SSP RK version
        //! Stage 1
    compute_residual(0,0);
    
    tmp_vec = stages[0];
    tmp_vec *= (delta_t)*(1./7.);
    u_i[1] = tmp_vec;
    u_i[1] += u_i[0];
    
    applyLimiters(u_i[1]);
    
        //! Stage 2
    compute_residual(1,1);
    tmp_vec = stages[1];
    tmp_vec *= (delta_t)*(3./16.);
    u_i[2] = tmp_vec;
    tmp_vec = u_i[0];
    u_i[2] += tmp_vec;
    
    applyLimiters(u_i[2]);
    
        //! Stage 3
    compute_residual(2,2);
    tmp_vec = stages[2];
    tmp_vec *= (delta_t)*(1./3.);
    u_i[3] = tmp_vec;
    tmp_vec = u_i[0];
    u_i[3] += tmp_vec;
    
    applyLimiters(u_i[3]);
    
        //! Stage 4
    compute_residual(3,3);
    tmp_vec = stages[3];
    tmp_vec *= (delta_t)*(2./3.);
    u_i[4] = tmp_vec;
    tmp_vec = u_i[0];
    u_i[4] += tmp_vec;
    
    applyLimiters(u_i[4]);
    
        //! Stage 5
    compute_residual(4,4);
    tmp_vec = stages[4];
    tmp_vec *= (delta_t)*(3./4.);
    u_i[5] = tmp_vec;
    tmp_vec = stages[0];
    tmp_vec *= (delta_t)*(1./4.);
    u_i[5] += tmp_vec;
    tmp_vec = u_i[0];
    u_i[5] += tmp_vec;
    
    applyLimiters(u_i[5]);
    
        //! Final solution for the timestep
    *(this->current_solution) = u_i[5];
    
    cur_time = old_time + delta_t;
    
}

template <int dim>
void TimeDiscretization<dim>::advance_SSP_5_4(void)
{
	ConstraintMatrix constraints;

    n_timestep++;
    
    double old_time = cur_time;
    
    if(n_timestep % 1000 == 0)
    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
              << "Global time = " << cur_time << sc::nl;
    
//    applyLimiters(*(this->current_solution));
//    current_solution->update_ghost_values();
    
        //! Set the old_solution equation to the current
        //! The "current" solution is then the initial guess
        //! for the Newton iterations
    set_old_from_current();
    
        //! Create temp vector
    ParVec &tmp_vec = stages[5];
    
    u_i[0] = *(this->current_solution);
//    u_i[0].update_ghost_values();
    
//    std::cout<<"\nu_i[0] = "<<u_i[0]<<std::endl;
//    std::cout<<"\nstages = "<<stages[0]<<std::endl;
//    getchar();
//    
//    std::cout<<"\nl2 norm of solution = "<<u_i[0].l2_norm();
    
        //!  Note:  stages[j] = RHS(u_i[j])
    
        //! Stage 1
    compute_residual(0,0);
    
    tmp_vec = stages[0];
    tmp_vec *= (delta_t)*(0.391752226571890);
    u_i[1] = tmp_vec;
    u_i[1] += u_i[0];
    
//    applyLimiters(u_i[1]);
//    stages[0].update_ghost_values();
//    u_i[1].update_ghost_values();
    
//    std::cout<<"\nu_i[1] = "<<u_i[1]<<std::endl;
//    std::cout<<"\nstages = "<<stages[1]<<std::endl;
//    getchar();
//    std::cout<<"\nl2 norm of solution = "<<u_i[1].l2_norm();
    
    cur_time = old_time + 0.39175222700392*delta_t;
    
    VectorTools::interpolate_boundary_values(dh, 
										0,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    if(dim==1)
    	VectorTools::interpolate_boundary_values(dh, 
										1,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    constraints.close();
    constraints.distribute(u_i[1]);
    constraints.clear();
    
        //! Stage 2
    compute_residual(1,1);
    tmp_vec = stages[1];
    tmp_vec *= (delta_t)*(0.368410593050371);
    u_i[2] = tmp_vec;
    tmp_vec = u_i[1];
    tmp_vec *= 0.555629506348765;
    u_i[2] += tmp_vec;
    tmp_vec = u_i[0];
    tmp_vec *= 0.444370493651235;
    u_i[2] += tmp_vec;
    
//    applyLimiters(u_i[2]);
        //    stages[0].update_ghost_values();
//    u_i[2].update_ghost_values();
    
//    std::cout<<"\nl2 norm of solution = "<<u_i[2].l2_norm();
//    std::cout<<"\nu_i[2] = "<<u_i[2]<<std::endl;
//    std::cout<<"\nstages = "<<stages[2]<<std::endl;
//    getchar();
    
    cur_time = old_time + 0.58607968896780*delta_t;
    
        VectorTools::interpolate_boundary_values(dh, 
										0,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    if(dim==1)
    	VectorTools::interpolate_boundary_values(dh, 
										1,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    constraints.close();
    constraints.distribute(u_i[2]);
    constraints.clear();
    
        //! Stage 3
    compute_residual(2,2);
    tmp_vec = stages[2];
    tmp_vec *= (delta_t)*(0.251891774271694);
    u_i[3] = tmp_vec;
    tmp_vec = u_i[2];
    tmp_vec *= 0.379898148511597;
    u_i[3] += tmp_vec;
    tmp_vec = u_i[0];
    tmp_vec *= 0.620101851488403;
    u_i[3] += tmp_vec;
    
//    applyLimiters(u_i[3]);
        //    stages[0].update_ghost_values();
//    u_i[3].update_ghost_values();
    
//    std::cout<<"\nu_i[3] = "<<u_i[3]<<std::endl;
//    std::cout<<"\nstages = "<<stages[3]<<std::endl;
//    getchar();
//    std::cout<<"\nl2 norm of solution = "<<u_i[3].l2_norm();
    
    cur_time = old_time + 0.47454236302687*delta_t;
    
    VectorTools::interpolate_boundary_values(dh, 
										0,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    if(dim==1)
    	VectorTools::interpolate_boundary_values(dh, 
										1,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    constraints.close();
    constraints.distribute(u_i[3]);
    constraints.clear();
    
        //! Stage 4
    compute_residual(3,3);
    tmp_vec = stages[3];
    tmp_vec *= (delta_t)*(0.544974750228521);
    u_i[4] = tmp_vec;
    tmp_vec = u_i[3];
    tmp_vec *= 0.821920045606868;
    u_i[4] += tmp_vec;
    tmp_vec = u_i[0];
    tmp_vec *= 0.178079954393132;
    u_i[4] += tmp_vec;
    
//    applyLimiters(u_i[4]);
        //    stages[0].update_ghost_values();
//    u_i[4].update_ghost_values();
    
//    std::cout<<"\nl2 norm of solution = "<<u_i[4].l2_norm();
    
    cur_time = old_time + 0.93501063100924*delta_t;
    
    VectorTools::interpolate_boundary_values(dh, 
										0,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    if(dim==1)
    	VectorTools::interpolate_boundary_values(dh, 
										1,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    constraints.close();
    constraints.distribute(u_i[4]);
    constraints.clear();

        //! Stage 5
    compute_residual(4,4);
    tmp_vec = stages[4];
    tmp_vec *= (delta_t)*(0.226007483236906);
    u_i[5] = tmp_vec;
    tmp_vec = stages[3];
    tmp_vec *= (delta_t)*(0.063692468666290);
    u_i[5] += tmp_vec;
    tmp_vec = u_i[4];
    tmp_vec *= 0.386708617503269;
    u_i[5] += tmp_vec;
    tmp_vec = u_i[3];
    tmp_vec *= 0.096059710526147;
    u_i[5] += tmp_vec;
    tmp_vec = u_i[2];
    tmp_vec *= 0.517231671970585;
    u_i[5] += tmp_vec;

//    std::cout<<"\nu_i[5] = "<<u_i[5]<<std::endl;
//    std::cout<<"\nstages = "<<stages[5]<<std::endl;
//    getchar();
    
//    applyLimiters(u_i[5]);
        //    stages[0].update_ghost_values();
//    u_i[5].update_ghost_values();
    
    
        //! Note:  the code below counts ghost cells twice!  
//    const double locally_relevant_norm = u_i[5].l2_norm();
    
//    const double total_norm = std::sqrt(Utilities::MPI::sum (locally_relevant_norm, MPI_COMM_WORLD));
    
//    std::cout<<"\nl2 norm of solution = "<<total_norm<<std::endl;
    
    cur_time = old_time + delta_t;
    
        VectorTools::interpolate_boundary_values(dh, 
										0,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    if(dim==1)
    	VectorTools::interpolate_boundary_values(dh, 
										1,
    									ExactSolution<dim>(2*dim, cur_time),
    									constraints);
    									
    constraints.close();
    constraints.distribute(u_i[5]);
    constraints.clear();
    
        //! Final solution for the timestep
    *(this->current_solution) = u_i[5];
    
}//advance_SSP_5_4



