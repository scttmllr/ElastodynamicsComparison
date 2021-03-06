#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

//c++
#include <fstream>
#include <iostream>


namespace ContinuousGalerkin
{
using namespace dealii;

	//! Template function that can be used to suppress compiler warnings
template<class T> void unused( const T& ) { }

#include "./user_defined_constants.h"
#include "./ExactSolutionFunction.h"


/******************************************** 
 * ELASTIC PROBLEM
 ********************************************/
template <int dim>
class ElasticProblem
{
public:
	ElasticProblem (const int poly_order,
                    const bool lumped_mass,
                    std::fstream &time_file);
	~ElasticProblem ();
    
        // Separate run functions for each time integrator:
	void run_AvgAccel (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_BackwardEuler (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_Bathe (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_CDF (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_ExpGenAlpha (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_ForwardEuler (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_GenAlpha (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_HHT (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_LinAccel (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    void run_WBZ (std::string time_integrator, int nx, int ny=-1, int nz=-1);
    
    
	void compute_errors ();
	std::vector<double> L1_error, L2_error;
    std::vector<std::string> L1_names, L2_names;
	
        // To access the data for convergnece table:
    unsigned int n_dofs, n_cells;
    
private:
	
	void setup_system ();
	
	void assemble_implicit_system();
	void assemble_explicit_system(Vector<double> &U);
	
	void output_results (const unsigned int cycle, 
						std::string time_integrator) const;
	
	void create_grid (int nx, int ny=-1, int nz=-1);
	
	Triangulation<dim>   triangulation;
	DoFHandler<dim>      dof_handler;
	
	FESystem<dim>        fe;
	
	ConstraintMatrix     constraints;
	
	SparsityPattern      sparsity_pattern;
	
	SparseMatrix<double> system_matrix;
	SparseMatrix<double> consistent_mass_matrix;
	SparseMatrix<double> stiffness_matrix;
	SparseMatrix<double> inverse_system_matrix;
	
	// For lumped mass matrices, only need to store
	// the diagonals; hence we just use a vector
	Vector<double> lumped_mass_matrix;
	Vector<double> inverse_lumped_mass_matrix;
    
    // Vectors for the RHS of the linear system,
    // as well as solutions
    Vector<double> system_rhs;
    Vector<double> solution, old_solution,
                    old_velocity, old_acceleration;
    
        // Temp vector for doing linear combinations of Vectors
    Vector<double> linear_combo;
	
	// Also, keep track of the current time and the time spent evaluating
    // certain functions
    TimerOutput                      computing_timer;
    
    // Info for time stepping:
    double current_time;
    unsigned int n_timesteps;
	
	// Use Q1 mapping (straight edges) for everything.
	// It is the default, so we need not explicitly use it.
//	const MappingQ1<dim> mapping;
	
	const FEValuesExtractors::Vector  disp;
};


template <int dim>
ElasticProblem<dim>::ElasticProblem (const int poly_order,
                                     const bool lumped_mass,
                                     std::fstream &time_file)
:
dof_handler (triangulation),
fe (FE_Q<dim>(poly_order), dim),
computing_timer(time_file, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
disp(0)
{}

template <int dim>
ElasticProblem<dim>::~ElasticProblem ()
{
	dof_handler.clear ();
}

template <int dim>
void ElasticProblem<dim>::setup_system ()
{
	dof_handler.distribute_dofs (fe);
	
	constraints.clear ();
    //DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    
    // Set Dirichlet BCs on all boundaries
	VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
											ExactSolution<dim>(dim, current_time),
											constraints);
    
	VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              ExactSolution<dim>(dim, current_time),
                                              constraints);
    
	constraints.close ();
	
	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
	sparsity_pattern.copy_from(c_sparsity);
	
	
	system_matrix.reinit (sparsity_pattern);
    consistent_mass_matrix.reinit (sparsity_pattern);
    stiffness_matrix.reinit (sparsity_pattern);
    
    system_rhs.reinit(dof_handler.n_dofs());
    
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    old_velocity.reinit(dof_handler.n_dofs());
    old_acceleration.reinit(dof_handler.n_dofs());
    linear_combo.reinit(dof_handler.n_dofs());
    
    this->n_dofs = dof_handler.n_dofs();
    this->n_cells = triangulation.n_active_cells();
    
}//setup_system


template <int dim>
void ElasticProblem<dim>::assemble_implicit_system ()
{
    // Zero matrices and vectors
    consistent_mass_matrix = 0.;
    stiffness_matrix = 0.;
    system_rhs = 0.;
    
        // Create an identity tensor:
    Tensor<2,dim> Id;
	Id = 0.0;
	for(int d=0; d<dim; ++d)
		Id[d][d] = 1;

    QGauss<dim>  quadrature_formula(std::ceil(((2.0*fe.degree) +1)/2));

    FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values   |
                            update_gradients |
                            update_quadrature_points | 
                            update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_mass (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   cell_stiffness (dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    // Now we can begin with the loop
    // over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                            endc = dof_handler.end();

    double l=lambda(dim);
    double m=mu(dim);
    
    double trE;
    Tensor<2,dim> w_j, E, Stress;

    for (; cell!=endc; ++cell)
 	{
 		cell_mass = 0;
        cell_stiffness = 0;
 		
 		fe_values.reinit (cell);
 		
 		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
 		{
 			for (unsigned int i=0; i<dofs_per_cell; ++i)
 			{
 				for (unsigned int j=0; j<dofs_per_cell; ++j)
 				{
                        // Compute the stress from the shape function:
                        // Linearized strain tensor:
                    E = 0.5 * (fe_values[disp].gradient(j,q_point)
                               + transpose(fe_values[disp].gradient(j,q_point) ) );
                    
                    trE = 0.0;
                    for(int d=0; d<dim; ++d)
                        trE += E[d][d];

                    Stress = Id;
                    Stress *= (l*trE);
                    Stress += (2.0*m)*E;
                    
                        // Assemble the local matrices
 					cell_mass(i,j) += rho * fe_values[disp].value(i,q_point) *
                                    fe_values[disp].value(j,q_point) *
                                    fe_values.JxW(q_point);
 					
 					cell_stiffness(i,j) += scalar_product(fe_values[disp].gradient(i,q_point),
                                    Stress) * fe_values.JxW(q_point);
 				}//j
 			}//i
 		}//q_point
 		
    cell->get_dof_indices (local_dof_indices);

       // Assemble the local matrices into the global system:
    constraints.distribute_local_to_global(cell_mass, local_dof_indices, consistent_mass_matrix);

    constraints.distribute_local_to_global(cell_stiffness, local_dof_indices, stiffness_matrix);
 		
 	}//cell

 }//assemble_implicit_system
    
template <int dim>
void ElasticProblem<dim>::assemble_explicit_system (Vector<double> &U)
{
        // Zero matrices and vectors
    system_rhs = 0.;
    lumped_mass_matrix = 0.;
    
        // Create an identity tensor:
    Tensor<2,dim> Id;
    Id = 0.0;
    for(int d=0; d<dim; ++d)
        Id[d][d] = 1;
    
    QGauss<dim>  quadrature_formula(std::ceil(((2.0*fe.degree) +1)/2));
    
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   |
                             update_gradients |
                             update_quadrature_points |
                             update_JxW_values);
    
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    
    Vector<double> cell_lumped_mass(dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    
        // Now we can begin with the loop
        // over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    
    double l=lambda(dim);
    double m=mu(dim);
    
    double trE;
    Tensor<2,dim> E, Stress;
    std::vector<Tensor<2,dim> > grad_u(n_q_points);
    
    for (; cell!=endc; ++cell)
    {
        cell_lumped_mass = 0;
        cell_rhs = 0;
        
        fe_values.reinit (cell);
        
        fe_values[disp].get_function_gradients(U, grad_u);
        
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                        // Assemble the local matrices
                    cell_lumped_mass(i) += rho * fe_values[disp].value(i,q_point) *
                    fe_values[disp].value(j,q_point) *
                    fe_values.JxW(q_point);
                }//j
                
                
                    // Assemble cell RHS from stress:
                    // Compute the stress from the shape function:
                    // Linearized strain tensor:
                E = 0.5*(grad_u[q_point] + transpose(grad_u[q_point]));
                trE = 0.0;
                for(int d=0; d<dim; ++d)
                    trE += E[d][d];
                
                Stress = Id;
                Stress *= (l*trE);
                Stress += (2.0*m)*E;
                
                cell_rhs(i) -= scalar_product(fe_values[disp].gradient(i,q_point), Stress)
                * fe_values.JxW(q_point);
            }//i
        }//q_point
        
        cell->get_dof_indices (local_dof_indices);
        
            // Assemble the local matrices into the global system:
            //    constraints.distribute_local_to_global(cell_lumped_mass, local_dof_indices, lumped_mass_matrix);
            //    constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
        for(unsigned int i=0; i<dofs_per_cell; ++i)
        {
            lumped_mass_matrix(local_dof_indices[i]) += cell_lumped_mass[i];
            system_rhs(local_dof_indices[i]) += cell_rhs[i];
        }
        
        
    }//cell
    
}//assemble_explicit_system

template <int dim>
void ElasticProblem<dim>::output_results (const unsigned int cycle, std::string time_integrator) const
{
	std::string filename = "./output_d";
	filename += Utilities::int_to_string(dim,1);
	filename += "/";//solution";
//    filename += Utilities::int_to_string (cycle, 6);
	filename += time_integrator;
	filename += "_" + Utilities::int_to_string(triangulation.n_active_cells(),6);
	filename += "_" + Utilities::int_to_string(fe.degree,1);
	//filename += "_" + Utilities::int_to_string (cycle, 6);
	filename += ".vtu";
	
	std::ofstream output (filename.c_str());
	
	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);
	
	std::vector<std::string> solution_names(dim, "displacement");
	
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	interpretation (dim, DataComponentInterpretation::component_is_part_of_vector);

	data_out.add_data_vector (solution, solution_names,
							DataOut<dim,DoFHandler<dim> >::type_automatic, 
							interpretation);
	
	data_out.build_patches (fe.degree);
	data_out.write_vtu (output);
		
}//output_results

template <int dim>
void ElasticProblem<dim>::create_grid (int nx, int ny, int nz)
{
	if (dim == 1)
        {
            const Point<dim> LowerLeft (0.0),
            UpperRight (1.0);
            
                // Define the subdivisions in the x1 and x2 coordinates.
            std::vector<unsigned int> subdivisions(dim);
            subdivisions[0] =   nx;
            
            // false indicates all boundaries have boundary_id = 0
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      subdivisions,
                                                      LowerLeft,
                                                      UpperRight,
                                                      false);
        }//dim==1
    else if (dim == 2)
        {
        const Point<dim> LowerLeft (0.0, 0.0),
                        UpperRight (1.0, 1.0);
        
            // Define the subdivisions in the x1 and x2 coordinates.
        std::vector<unsigned int> subdivisions(dim);
        subdivisions[0] =   nx;
        subdivisions[1] =   ny;
        
        // false indicates all boundaries have boundary_id = 0
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  subdivisions,
                                                  LowerLeft,
                                                  UpperRight,
                                                  false);
        }//dim==2
	else if (dim == 3)
        {
            const Point<dim> LowerLeft (0.0, 0.0, 0.0),
            UpperRight (1.0, 1.0, 1.0);
            
                // Define the subdivisions in the x1 and x2 coordinates.
            std::vector<unsigned int> subdivisions(dim);
            subdivisions[0] =   nx;
            subdivisions[1] =   ny;
            subdivisions[2] =   nz;
            
            // false indicates all boundaries have boundary_id = 0
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      subdivisions,
                                                      LowerLeft,
                                                      UpperRight,
                                                      false);
        }//dim==3

}//create_grid

template <int dim>
void ElasticProblem<dim>::compute_errors(void)
{
        // The number of errors we compute depends on dimension:
    unsigned int n_comp = 0;
    
    if(dim==1)
    {
        n_comp = 4; // u,v,E,energy
        
        L1_error.resize(n_comp, 0.0);
        L1_names.resize(n_comp);
        
        L2_error.resize(n_comp, 0.0);
        L2_names.resize(n_comp);
        
        L1_names[0] = "L1_u_x";
        L1_names[1] = "L1_v_x";
        L1_names[2] = "L1_E_xx";
        L1_names[3] = "L1_energy";
        
        L2_names[0] = "L2_u_x";
        L2_names[1] = "L2_v_x";
        L2_names[2] = "L2_E_xx";
        L2_names[3] = "L2_energy";
    }
    else if (dim==2)
    {
        n_comp = 8; // ux, uy, vx, vy, Exx, Exy, Eyy, energy
        
        L1_error.resize(n_comp, 0.0);
        L1_names.resize(n_comp);
        
        L2_error.resize(n_comp, 0.0);
        L2_names.resize(n_comp);
        
        L1_names[0] = "L1_u_x";
        L1_names[1] = "L1_u_y";
        L1_names[2] = "L1_v_x";
        L1_names[3] = "L1_v_y";
        L1_names[4] = "L1_E_xx";
        L1_names[5] = "L1_E_yy";
        L1_names[6] = "L1_E_xy";
        L1_names[7] = "L1_energy";
        
        L2_names[0] = "L2_u_x";
        L2_names[1] = "L2_u_y";
        L2_names[2] = "L2_v_x";
        L2_names[3] = "L2_v_y";
        L2_names[4] = "L2_E_xx";
        L2_names[5] = "L2_E_yy";
        L2_names[6] = "L2_E_xy";
        L2_names[7] = "L2_energy";
    }
    
        // Create an identity tensor:
    Tensor<2,dim> Id;
	Id = 0.0;
	for(int d=0; d<dim; ++d)
		Id[d][d] = 1;
    
    QGauss<dim>  quadrature_formula(std::ceil(((3.0*fe.degree) +1)/2));
    
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   |
                             update_gradients |
                             update_quadrature_points |
                             update_JxW_values);
    
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    
    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
    
        // Now we can begin with the loop
        // over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    
    double l=lambda(dim);
    double m=mu(dim);
    
    double trE;
    Tensor<2,dim> E, Stress;
    
        // Numerical solution values
    std::vector<Tensor<1,dim> > u(n_q_points), v(n_q_points);
    std::vector<Tensor<2,dim> > grad_u(n_q_points);
    
        // Exact solution:  prefaced with "e_" for "exact"
    ExactSolution<dim> e_soln(dim, current_time);
    Tensor<1,dim> e_u, e_v;
    Tensor<2,dim> e_E, e_S;
    
        // Energies:
    double energy_h, e_energy;
    
    for (; cell!=endc; ++cell)
 	{
 		fe_values.reinit (cell);
        
        fe_values[disp].get_function_values(solution, u);
        fe_values[disp].get_function_values(old_velocity, v);
        fe_values[disp].get_function_gradients(solution, grad_u);
 		
 		for (unsigned int q=0; q<n_q_points; ++q)
 		{
            double JxW = fe_values.JxW(q);
            Point<dim> point = fe_values.quadrature_point(q);
            
            if(dim==1)
            {
                e_u[0] = e_soln.value(point, 0);
                e_v[0] = e_soln.value(point, 1);
                e_E[0][0] = e_soln.value(point, 2);
            }
            else if (dim==2)
            {
                e_u[0] = e_soln.value(point, 0);
                e_u[1] = e_soln.value(point, 1);
                e_v[0] = e_soln.value(point, 2);
                e_v[1] = e_soln.value(point, 3);
                e_E[0][0] = e_soln.value(point, 4);
                e_E[1][1] = e_soln.value(point, 5);
                e_E[0][1] = e_soln.value(point, 6);
                e_E[1][0] = e_E[0][1];
            }
            
            E = 0.5*(grad_u[q] + transpose(grad_u[q]));
            trE = 0.0;
            for(int d=0; d<dim; ++d)
                trE += E[d][d];
            
            Stress = Id;
            Stress *= (l*trE);
            Stress += (2.0*m)*E;
            
            trE = 0.0;
            for(int d=0; d<dim; ++d)
                trE += e_E[d][d];
            
            e_S = Id;
            e_S *= (l*trE);
            e_S += (2.0*m)*e_E;
            
            energy_h = rho*(v[q]*v[q]) + scalar_product(E,Stress);
            e_energy = rho*(e_v*e_v) + scalar_product(e_E, e_S);
            
            if(dim==1)
            {
                L1_error[0] += std::abs(u[q][0] - e_u[0])*JxW;
                L1_error[1] += std::abs(v[q][0] - e_v[0])*JxW;
                L1_error[2] += std::abs(E[0][0] - e_E[0][0])*JxW;
                L1_error[3] += std::abs(energy_h - e_energy)*JxW;
                
                L2_error[0] += (u[q][0] - e_u[0])*(u[q][0] - e_u[0])*JxW;
                L2_error[1] += (v[q][0] - e_v[0])*(v[q][0] - e_v[0])*JxW;
                L2_error[2] += (E[0][0] - e_E[0][0])*(E[0][0] - e_E[0][0])*JxW;
                L2_error[3] += (energy_h - e_energy)*(energy_h - e_energy)*JxW;
            }
            else if (dim==2)
            {
                L1_error[0] += std::abs(u[q][0] - e_u[0])*JxW;
                L1_error[1] += std::abs(u[q][1] - e_u[1])*JxW;
                L1_error[2] += std::abs(v[q][0] - e_v[0])*JxW;
                L1_error[3] += std::abs(v[q][1] - e_v[1])*JxW;
                L1_error[4] += std::abs(E[0][0] - e_E[0][0])*JxW;
                L1_error[5] += std::abs(E[1][1] - e_E[1][1])*JxW;
                L1_error[6] += std::abs(E[0][1] - e_E[0][1])*JxW;
                L1_error[7] += std::abs(energy_h - e_energy)*JxW;
                
                L2_error[0] += (u[q][0] - e_u[0])*(u[q][0] - e_u[0])*JxW;
                L2_error[1] += (u[q][1] - e_u[1])*(u[q][1] - e_u[1])*JxW;
                L2_error[2] += (v[q][0] - e_v[0])*(v[q][0] - e_v[0])*JxW;
                L2_error[3] += (v[q][1] - e_v[1])*(v[q][1] - e_v[1])*JxW;
                L2_error[4] += (E[0][0] - e_E[0][0])*(E[0][0] - e_E[0][0])*JxW;
                L2_error[5] += (E[1][1] - e_E[1][1])*(E[1][1] - e_E[1][1])*JxW;
                L2_error[6] += (E[0][1] - e_E[0][1])*(E[0][1] - e_E[0][1])*JxW;
                L2_error[7] += (energy_h - e_energy)*(energy_h - e_energy)*JxW;
            }

 		}//q_point
    }//cell
    
    for(unsigned int i=0; i<L2_error.size(); ++i)
        L2_error[i] = std::sqrt(L2_error[i]);
    
}//compute_errors
    
template <int dim>
void ElasticProblem<dim>::run_AvgAccel (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.25;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
        // Set parameters for the generalized alpha method:
    double rhoInf = 0.25;
    double a_f = rhoInf/(1.+rhoInf);
    double a_m = (2.*rhoInf-1.)/(1.+rhoInf);
    double beta = (1.-a_m+a_f)*(1.-a_m+a_f)/4.;
    double gamma = 0.5 - a_m + a_f;
    
    a_f=0.;
    a_m=0.;
    beta = 0.25;
    gamma = 0.5;
    
        // Store some coefficients:
    double a1 = (1.-a_m)/(beta*delta_t*delta_t);
    double a2 = (1.-a_f);
    double a3 = (1.-a_m)/(beta*delta_t);
    double a4 = (1.-a_m-2.*beta)/(2.*beta);
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionSecondTimeDerivative<dim>(dim, current_time),
                             old_acceleration);
    
    constraints.distribute (solution);
    constraints.distribute (old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_implicit_system();
        system_matrix = 0.;
        system_matrix.add(a1,consistent_mass_matrix);
        system_matrix.add(a2,stiffness_matrix);
        computing_timer.exit_section();
        
        computing_timer.enter_section("Compute RHS");
        
        linear_combo = 0.0;
        linear_combo.add(a1, old_solution);
        linear_combo.add(a3, old_velocity);
        linear_combo.add(a4, old_acceleration);
        
        system_rhs = 0.0;
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        
            //        linear_combo = 0.0;
            //        stiffness_matrix.vmult(linear_combo, old_solution);
            //        linear_combo *= (-1.*a_f);
            //        system_rhs += linear_combo;
        
        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        {
            SparseDirectUMFPACK directSolver;
            directSolver.initialize (system_matrix);
            directSolver.vmult (solution, system_rhs);
            constraints.distribute (solution);
        }
        
        computing_timer.exit_section();//Linear solve
        
        
            // Set the "old" velocity and acceleration
            // for use in the next timestep
        linear_combo = old_acceleration;
        
        old_acceleration = solution;// u^{N+1}
        old_acceleration -= old_solution;// u^{N}
        old_acceleration.add(-delta_t, old_velocity);// udot^{N}
        old_acceleration *= (2.*inv_dt*inv_dt);
        old_acceleration.add(2.*beta-1.,linear_combo);// uddot^N
        old_acceleration *= 1./(2.*beta);
        
        old_velocity.add(delta_t*(1-gamma), linear_combo);
        old_velocity.add(delta_t*gamma, old_acceleration);
        constraints.distribute (old_velocity);
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();
    
    computing_timer.print_summary();
    
        // Output the results
        //output_results (n_timesteps, time_integrator);
    
}

template <int dim>
void ElasticProblem<dim>::run_BackwardEuler (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
	create_grid(nx, ny, nz);
    
	std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells()
            << std::endl;
	
    computing_timer.enter_section("Setup DOF systems");
	setup_system ();
    computing_timer.exit_section();
	
	std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
    
        // Set time stepping parameters:
				
	//output_results(0, time_integrator);
	
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.25;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
	VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    constraints.distribute (solution);
    constraints.distribute (old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
        // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_implicit_system();
        computing_timer.exit_section();
        
        computing_timer.enter_section("Compute RHS");
        system_rhs = 0.0;
        linear_combo = old_solution;
        linear_combo *= inv_dt;
        linear_combo += old_velocity;
        
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        system_rhs *= inv_dt;
        
        system_matrix = 0.;
        system_matrix.add(inv_dt*inv_dt,consistent_mass_matrix);
        system_matrix.add(1.0,stiffness_matrix);
        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        SparseDirectUMFPACK directSolver;
        directSolver.initialize (system_matrix);
        directSolver.vmult (solution, system_rhs);
        
        constraints.distribute (solution);
        computing_timer.exit_section();//Linear solve
        
        old_velocity = solution;
        old_velocity -= old_solution;
        old_velocity *= inv_dt;
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();

    computing_timer.print_summary();	
        // Output the results
	//output_results (n_timesteps, time_integrator);
	
}
    
template <int dim>
void ElasticProblem<dim>::run_Bathe (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.25;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // We need displacement and velocity vectors at the mid-point
        // of the timestep for Bathe's integrator:
    Vector<double> mid_U(n_dofs), mid_Udot(n_dofs);
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionSecondTimeDerivative<dim>(dim, current_time),
                             old_acceleration);
    
    constraints.distribute (solution);
    constraints.distribute (old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t/2.;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        mid_U = 0.;
        mid_Udot = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_implicit_system();
        system_matrix = 0.;
        system_matrix.add(16.*inv_dt*inv_dt,consistent_mass_matrix);
        system_matrix.add(1.0,stiffness_matrix);
        computing_timer.exit_section();
        
        computing_timer.enter_section("Compute RHS");
        system_rhs = 0.0;
        linear_combo = 0.0;
        
        linear_combo.add(16.*inv_dt*inv_dt, old_solution);
        linear_combo.add(8.*inv_dt, old_velocity);
        linear_combo.add(1., old_acceleration);
        
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        
        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        {
            SparseDirectUMFPACK directSolver;
            directSolver.initialize (system_matrix);
            directSolver.vmult (mid_U, system_rhs);
        }
        
        constraints.distribute (mid_U);
        computing_timer.exit_section();//Linear solve
        
            // Compute velocity at middle of timestep:
        mid_Udot = 0.;
        mid_Udot.add(-1., old_velocity);
        mid_Udot.add(4.*inv_dt, mid_U);
        mid_Udot.add(-4.*inv_dt, old_solution);
        constraints.distribute (mid_Udot);
        
            // Step 2:  BDF2
        current_time += delta_t/2.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_implicit_system();
        system_matrix = 0.;
        system_matrix.add(9.*inv_dt*inv_dt,consistent_mass_matrix);
        system_matrix.add(1.0,stiffness_matrix);
        computing_timer.exit_section();
        
        computing_timer.enter_section("Compute RHS");
        system_rhs = 0.0;
        linear_combo = 0.0;
        
        linear_combo.add(12.*inv_dt*inv_dt, mid_U);
        linear_combo.add(-3.*inv_dt*inv_dt, old_solution);
        linear_combo.add(4.*inv_dt, mid_Udot);
        linear_combo.add(-1.*inv_dt, old_velocity);
        
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        
        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        {
            SparseDirectUMFPACK directSolver;
            directSolver.initialize (system_matrix);
            directSolver.vmult (solution, system_rhs);
        }
        
        constraints.distribute (solution);
        computing_timer.exit_section();//Linear solve
        
            // Set the "old" velocity and acceleration
            // for use in the next timestep
            // Note:  need to keep the old_velocity for
            // calculation of the old_acceleration:
        linear_combo = old_velocity;
        
        old_velocity = 0.;
        old_velocity.add(3.*inv_dt, solution);// u^{N+1}
        old_velocity.add(-4.*inv_dt, mid_U);// u^{N+1/2}
        old_velocity.add(inv_dt, old_solution);// u^N
        constraints.distribute (old_velocity);
        
        old_acceleration = 0.;
        old_acceleration.add(3.*inv_dt, old_velocity);// udot^{N+1}
        old_acceleration.add(-4.*inv_dt, mid_Udot);// udot^{N+1/2}
        old_acceleration.add(inv_dt, linear_combo);// udot^N
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();
    
    computing_timer.print_summary();	
        // Output the results
        //output_results (n_timesteps, time_integrator);
}//Bathe
    
template <int dim>
void ElasticProblem<dim>::run_CDF (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.1;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionSecondTimeDerivative<dim>(dim, current_time),
                             old_acceleration);
    
    constraints.distribute(solution);
    constraints.distribute(old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
            // Compute solution at t^{n+1}
        solution.add(delta_t, old_velocity);
        solution.add(0.5*delta_t*delta_t, old_acceleration);
        constraints.distribute (solution);
        
        computing_timer.enter_section("Assemble matrices");
        assemble_explicit_system(solution);
        computing_timer.exit_section();
        
            //        computing_timer.enter_section("Compute RHS");
        
            //        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        
            // system_rhs now = -K*U^{n+1}
            // old_solution vector is not needed, so we use
            // it as temporary storage
        for(unsigned int dof=0; dof<dof_handler.n_dofs(); ++dof)
        {
                // old_solution = \ddot{U}^{n+1}
            old_solution[dof] = -1.*system_rhs[dof]/lumped_mass_matrix[dof];
        }
        
            // Update the velocity:
        old_velocity.add(0.5*delta_t, old_acceleration);
        old_velocity.add(0.5*delta_t, old_solution);
        constraints.distribute (old_velocity);
        
            // And now store the old_acceleration properly
        old_acceleration = old_solution;
        
        computing_timer.exit_section();//Linear solve
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();
    
    computing_timer.print_summary();
        // Output the results
        //	output_results (n_timesteps, time_integrator);
    
}//CDF
    
template <int dim>
void ElasticProblem<dim>::run_ExpGenAlpha (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.1;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
        // Set parameters for the generalized alpha method:
    double rhoB = 0.25;
    double a_m = (2.*rhoB-1.)/(1.+rhoB);
    double beta = (5.-3.*rhoB)/( (1.+rhoB)*(1.+rhoB)*(2.-rhoB) );
    double gamma = 1.5 - a_m;
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    constraints.distribute(solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionSecondTimeDerivative<dim>(dim, current_time),
                             old_acceleration);
    
    constraints.distribute(solution);
    constraints.distribute(old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_explicit_system(old_solution);
        computing_timer.exit_section();
        
            //        computing_timer.enter_section("Compute RHS");
        
            //        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        
            // First compute \ddot{U}^{n+1} and store in
            // the temp `linear_combo`
        for(unsigned int dof=0; dof<dof_handler.n_dofs(); ++dof)
        {
            linear_combo[dof] = system_rhs[dof]/lumped_mass_matrix[dof];
        }
        
        linear_combo.add(-a_m, old_acceleration);
        linear_combo *= 1./(1.-a_m);
        
            // Next compute U^{n+1}
        solution = old_solution;
        solution.add(delta_t, old_velocity);
        
        double tmp = 0.5*delta_t*delta_t*(1.-2.*beta);
        solution.add(tmp, old_acceleration);
        tmp = 0.5*delta_t*delta_t*(2.*beta);
        solution.add(tmp, linear_combo);
        
        constraints.distribute(solution);
        
            // Compute \dot{U}^{n+1}
        old_velocity.add(delta_t*(1.-gamma), old_acceleration);
        old_velocity.add(delta_t*gamma, linear_combo);
        
        constraints.distribute(old_velocity);
        
            // Store the acceleration:
        old_acceleration = linear_combo;
        
        computing_timer.exit_section();//Linear solve
        
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    
    compute_errors();
    
    computing_timer.print_summary();
        // Output the results
        //	output_results (n_timesteps, time_integrator);
}//ExpGenAlpha
    
template <int dim>
void ElasticProblem<dim>::run_ForwardEuler (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.1;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    constraints.distribute(solution);
    constraints.distribute(old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_explicit_system(old_solution);
        computing_timer.exit_section();
        
            //        computing_timer.enter_section("Compute RHS");
        
            //        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        
        system_rhs *= delta_t*delta_t;
        
        for(unsigned int dof=0; dof<dof_handler.n_dofs(); ++dof)
        {
            solution[dof] = system_rhs[dof]/lumped_mass_matrix[dof];
        }
        
        solution += old_solution;
        solution.add(delta_t, old_velocity);
        
        constraints.distribute (solution);
        computing_timer.exit_section();//Linear solve
        
        old_velocity = solution;
        old_velocity -= old_solution;
        old_velocity *= inv_dt;
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();
    
    computing_timer.print_summary();
    
        // Output the results
        //	output_results (n_timesteps, time_integrator);
    
}//ForwardEuler
    
template <int dim>
void ElasticProblem<dim>::run_GenAlpha (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.25;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
        // Set parameters for the generalized alpha method:
    double rhoInf = 0.25;
    double a_f = rhoInf/(1.+rhoInf);
    double a_m = (2.*rhoInf-1.)/(1.+rhoInf);
    double beta = (1.-a_m+a_f)*(1.-a_m+a_f)/4.;
    double gamma = 0.5 - a_m + a_f;
    
        // Store some coefficients:
    double a1 = (1.-a_m)/(beta*delta_t*delta_t);
    double a2 = (1.-a_f);
    double a3 = (1.-a_m)/(beta*delta_t);
    double a4 = (1.-a_m-2.*beta)/(2.*beta);
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionSecondTimeDerivative<dim>(dim, current_time),
                             old_acceleration);
    
    constraints.distribute (solution);
    constraints.distribute (old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_implicit_system();
        system_matrix = 0.;
        system_matrix.add(a1,consistent_mass_matrix);
        system_matrix.add(a2,stiffness_matrix);
        computing_timer.exit_section();
        
        computing_timer.enter_section("Compute RHS");
        
        linear_combo = 0.0;
        linear_combo.add(a1, old_solution);
        linear_combo.add(a3, old_velocity);
        linear_combo.add(a4, old_acceleration);
        
        system_rhs = 0.0;
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        
        linear_combo = 0.0;
        stiffness_matrix.vmult(linear_combo, old_solution);
        linear_combo *= (-1.*a_f);
        system_rhs += linear_combo;
        
        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        {
            SparseDirectUMFPACK directSolver;
            directSolver.initialize (system_matrix);
            directSolver.vmult (solution, system_rhs);
            constraints.distribute (solution);
        }
        
        computing_timer.exit_section();//Linear solve
        
        
            // Set the "old" velocity and acceleration
            // for use in the next timestep
        linear_combo = old_acceleration;
        
        old_acceleration = solution;// u^{N+1}
        old_acceleration -= old_solution;// u^{N}
        old_acceleration.add(-delta_t, old_velocity);// udot^{N}
        old_acceleration *= (2.*inv_dt*inv_dt);
        old_acceleration.add(2.*beta-1.,linear_combo);// uddot^N
        old_acceleration *= 1./(2.*beta);
        
        old_velocity.add(delta_t*(1-gamma), linear_combo);
        old_velocity.add(delta_t*gamma, old_acceleration);
        constraints.distribute (old_velocity);
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();
    
    computing_timer.print_summary();
        // Output the results
        //output_results (n_timesteps, time_integrator);
    
}//GenAlpha
    
template <int dim>
void ElasticProblem<dim>::run_HHT (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.25;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
        // Set parameters for the generalized alpha method:
    double rhoInf = 0.5;
    double a_f = rhoInf/(1.+rhoInf);
    double a_m = (2.*rhoInf-1.)/(1.+rhoInf);
    double beta = (1.-a_m+a_f)*(1.-a_m+a_f)/4.;
    double gamma = 0.5 - a_m + a_f;
    
        // Store some coefficients:
    double a1 = (1.-a_m)/(beta*delta_t*delta_t);
    double a2 = (1.-a_f);
    double a3 = (1.-a_m)/(beta*delta_t);
    double a4 = (1.-a_m-2.*beta)/(2.*beta);
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionSecondTimeDerivative<dim>(dim, current_time),
                             old_acceleration);
    
    constraints.distribute (solution);
    constraints.distribute (old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_implicit_system();
        system_matrix = 0.;
        system_matrix.add(a1,consistent_mass_matrix);
        system_matrix.add(a2,stiffness_matrix);
        computing_timer.exit_section();
        
        computing_timer.enter_section("Compute RHS");
        
        linear_combo = 0.0;
        linear_combo.add(a1, old_solution);
        linear_combo.add(a3, old_velocity);
        linear_combo.add(a4, old_acceleration);
        
        system_rhs = 0.0;
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        
        linear_combo = 0.0;
        stiffness_matrix.vmult(linear_combo, old_solution);
        linear_combo *= (-1.*a_f);
        system_rhs += linear_combo;
        
        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        {
            SparseDirectUMFPACK directSolver;
            directSolver.initialize (system_matrix);
            directSolver.vmult (solution, system_rhs);
            constraints.distribute (solution);
        }
        
        computing_timer.exit_section();//Linear solve
        
        
            // Set the "old" velocity and acceleration
            // for use in the next timestep
        linear_combo = old_acceleration;
        
        old_acceleration = solution;// u^{N+1}
        old_acceleration -= old_solution;// u^{N}
        old_acceleration.add(-delta_t, old_velocity);// udot^{N}
        old_acceleration *= (2.*inv_dt*inv_dt);
        old_acceleration.add(2.*beta-1.,linear_combo);// uddot^N
        old_acceleration *= 1./(2.*beta);
        
        old_velocity.add(delta_t*(1-gamma), linear_combo);
        old_velocity.add(delta_t*gamma, old_acceleration);
        constraints.distribute (old_velocity);
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();
    
    computing_timer.print_summary();
        // Output the results
        //output_results (n_timesteps, time_integrator);
    
}//HHT
    
template <int dim>
void ElasticProblem<dim>::run_LinAccel (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.25;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
        // Set parameters for the generalized alpha method:
    double rhoInf = 0.25;
    double a_f = rhoInf/(1.+rhoInf);
    double a_m = (2.*rhoInf-1.)/(1.+rhoInf);
    double beta = (1.-a_m+a_f)*(1.-a_m+a_f)/4.;
    double gamma = 0.5 - a_m + a_f;
    
    a_f=0.;
    a_m=0.;
    beta=1./6.;
    gamma = 0.5;
    
        // Store some coefficients:
    double a1 = (1.-a_m)/(beta*delta_t*delta_t);
    double a2 = (1.-a_f);
    double a3 = (1.-a_m)/(beta*delta_t);
    double a4 = (1.-a_m-2.*beta)/(2.*beta);
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionSecondTimeDerivative<dim>(dim, current_time),
                             old_acceleration);
    
    constraints.distribute (solution);
    constraints.distribute (old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_implicit_system();
        system_matrix = 0.;
        system_matrix.add(a1,consistent_mass_matrix);
        system_matrix.add(a2,stiffness_matrix);
        computing_timer.exit_section();
        
        computing_timer.enter_section("Compute RHS");
        
        linear_combo = 0.0;
        linear_combo.add(a1, old_solution);
        linear_combo.add(a3, old_velocity);
        linear_combo.add(a4, old_acceleration);
        
        system_rhs = 0.0;
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        
            //        linear_combo = 0.0;
            //        stiffness_matrix.vmult(linear_combo, old_solution);
            //        linear_combo *= (-1.*a_f);
            //        system_rhs += linear_combo;
        
        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        {
            SparseDirectUMFPACK directSolver;
            directSolver.initialize (system_matrix);
            directSolver.vmult (solution, system_rhs);
            constraints.distribute (solution);
        }
        
        computing_timer.exit_section();//Linear solve
        
        
            // Set the "old" velocity and acceleration
            // for use in the next timestep
        linear_combo = old_acceleration;
        
        old_acceleration = solution;// u^{N+1}
        old_acceleration -= old_solution;// u^{N}
        old_acceleration.add(-delta_t, old_velocity);// udot^{N}
        old_acceleration *= (2.*inv_dt*inv_dt);
        old_acceleration.add(2.*beta-1.,linear_combo);// uddot^N
        old_acceleration *= 1./(2.*beta);
        
        old_velocity.add(delta_t*(1-gamma), linear_combo);
        old_velocity.add(delta_t*gamma, old_acceleration);
        constraints.distribute (old_velocity);
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();
    
    computing_timer.print_summary();
    
        // Output the results
        //output_results (n_timesteps, time_integrator);
    
}//LinAccel
    
template <int dim>
void ElasticProblem<dim>::run_WBZ (std::string time_integrator, int nx, int ny, int nz)
{
    
    computing_timer.enter_section("Total run time");
    
    create_grid(nx, ny, nz);
    
    std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;
    
    computing_timer.enter_section("Setup DOF systems");
    setup_system ();
    computing_timer.exit_section();
    
    std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
        // Set time stepping parameters:
    
        //output_results(0, time_integrator);
    
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // Set time step size based on a constant CFL:
    double cfl = 0.25;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
        // Set parameters for the generalized alpha method:
    double rhoInf = 0.;
    double a_f = rhoInf/(1.+rhoInf);
    double a_m = (2.*rhoInf-1.)/(1.+rhoInf);
    double beta = (1.-a_m+a_f)*(1.-a_m+a_f)/4.;
    double gamma = 0.5 - a_m + a_f;
    
        // Store some coefficients:
    double a1 = (1.-a_m)/(beta*delta_t*delta_t);
    double a2 = (1.-a_f);
    double a3 = (1.-a_m)/(beta*delta_t);
    double a4 = (1.-a_m-2.*beta)/(2.*beta);
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    std::cout << std::endl;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionSecondTimeDerivative<dim>(dim, current_time),
                             old_acceleration);
    
    constraints.distribute (solution);
    constraints.distribute (old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
            // Set old solution from the current
        old_solution = solution;
        solution = 0.;
        
        computing_timer.enter_section("Assemble matrices");
        assemble_implicit_system();
        system_matrix = 0.;
        system_matrix.add(a1,consistent_mass_matrix);
        system_matrix.add(a2,stiffness_matrix);
        computing_timer.exit_section();
        
        computing_timer.enter_section("Compute RHS");
        
        linear_combo = 0.0;
        linear_combo.add(a1, old_solution);
        linear_combo.add(a3, old_velocity);
        linear_combo.add(a4, old_acceleration);
        
        system_rhs = 0.0;
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        
        linear_combo = 0.0;
        stiffness_matrix.vmult(linear_combo, old_solution);
        linear_combo *= (-1.*a_f);
        system_rhs += linear_combo;
        
        computing_timer.exit_section();//Compute RHS
        
        computing_timer.enter_section("Linear solve");
        {
            SparseDirectUMFPACK directSolver;
            directSolver.initialize (system_matrix);
            directSolver.vmult (solution, system_rhs);
            constraints.distribute (solution);
        }
        
        computing_timer.exit_section();//Linear solve
        
        
            // Set the "old" velocity and acceleration
            // for use in the next timestep
        linear_combo = old_acceleration;
        
        old_acceleration = solution;// u^{N+1}
        old_acceleration -= old_solution;// u^{N}
        old_acceleration.add(-delta_t, old_velocity);// udot^{N}
        old_acceleration *= (2.*inv_dt*inv_dt);
        old_acceleration.add(2.*beta-1.,linear_combo);// uddot^N
        old_acceleration *= 1./(2.*beta);
        
        old_velocity.add(delta_t*(1-gamma), linear_combo);
        old_velocity.add(delta_t*gamma, old_acceleration);
        constraints.distribute (old_velocity);
    }
    
        // Total run time section
    computing_timer.exit_section();
    
        // Compute and output the errors:
    compute_errors();
    
    computing_timer.print_summary();
        // Output the results
        //output_results (n_timesteps, time_integrator);
    
}//WBZ
    
}//namespace

/***********************************
 * MAIN
 ***********************************/
int main (int argc, char* argv[])
{
    try
    {
            // Parse input:
    std::string prm_file_name;
    
    if(argc < 2)
    {
        dealii::deallog << "No Parameter File Specified!!!" << std::endl;
        dealii::deallog << "Using Default Filename: Parameters.prm" << std::endl;
        prm_file_name = "parameters.prm";
    }
    else
    {
        prm_file_name = argv[1];
    }
        
        // Open the file:
    std::fstream input_stream;
    input_stream.open(prm_file_name.c_str(), std::ios::in);
        
            // Read the data:
        std::vector<std::string> input_strings(4);
        std::vector<int>         input_integers(4);
        
        for(int i=0; i<4; ++i)
        {
            input_stream >> input_strings[i];
            input_stream >> input_integers[i];
        }
        
        // Close the file:
    input_stream.close();
        
        
//        for(int i=0; i<4; ++i)
//        {
//            std::cout<<"\n"<< input_strings[i];
//            std::cout<<"\n"<< input_integers[i]<<std::endl;
//        }
//        getchar();
        
        const unsigned int scheme = input_integers[0];
        const unsigned int dim = input_integers[1];
        const unsigned int p = input_integers[2];
        const unsigned int h = input_integers[3];
        
        
            // Strings for output
        std::vector<std::string> time_integrator(10);
    
        time_integrator[0] = "AvgAccel";
        time_integrator[1] = "BackwardEuler";
        time_integrator[2] = "Bathe";
        time_integrator[3] = "CDF";
        time_integrator[4] = "ExpGenAlpha";
        time_integrator[5] = "ForwardEuler";
        time_integrator[6] = "GenAlpha";
        time_integrator[7] = "HHT";
        time_integrator[8] = "LinAccel";
        time_integrator[9] = "WBZ";
        
        std::string sh = dealii::Utilities::int_to_string(h,4);
        std::string sp = dealii::Utilities::int_to_string(p,4);
        std::string sd = dealii::Utilities::int_to_string(dim,4);

    std::string fileName = "./" + time_integrator[scheme] + "_Timing_d" + sd + "_p" + sp + "_h" + sh + ".dat";
        
    std::fstream timing_stream;
    timing_stream.open(fileName.c_str(), std::ios::out);
        
            // To access the data for output
        unsigned int n_dofs, n_cells;
        
            // Giant hack!
        if(dim==1)
        {
        ContinuousGalerkin::ElasticProblem<1> ed_problem(p, false, timing_stream);
          
            switch(scheme)
            {
                case 0:
                    ed_problem.run_AvgAccel(time_integrator[scheme], h);
                    break;
                case 1:
                    ed_problem.run_BackwardEuler(time_integrator[scheme], h);
                    break;
                case 2:
                    ed_problem.run_Bathe(time_integrator[scheme], h);
                    break;
                case 3:
                    ed_problem.run_CDF(time_integrator[scheme], h);
                    break;
                case 4:
                    ed_problem.run_ExpGenAlpha(time_integrator[scheme], h);
                    break;
                case 5:
                    ed_problem.run_ForwardEuler(time_integrator[scheme], h);
                    break;
                case 6:
                    ed_problem.run_GenAlpha(time_integrator[scheme], h);
                    break;
                case 7:
                    ed_problem.run_HHT(time_integrator[scheme], h);
                    break;
                case 8:
                    ed_problem.run_LinAccel(time_integrator[scheme], h);
                    break;
                case 9:
                    ed_problem.run_WBZ(time_integrator[scheme], h);
                    break;
                default:
                    break;
                    
            }
            
            n_cells = ed_problem.n_cells;
            n_dofs = ed_problem.n_dofs;
            
            timing_stream.close();
            
            {
                    // Dump to a file:
                    // polynomial order
                    // nx
                    // cells
                    // dofs
                    // L1 errors
                    // L2 errors
                std::string fileName = "./" + time_integrator[scheme] + "Errors_d"+sd+"1_p"
                + sp + "_h" + sh + ".dat";
                std::fstream fp;
                fp.open(fileName.c_str(), std::ios::out);
                fp << p << std::endl;
                fp << h << std::endl;
                fp << n_cells << std::endl;
                fp << n_dofs << std::endl;
                
                fp.precision(16);
                
                for(unsigned int i=0; i<ed_problem.L1_error.size(); ++i)
                    fp << ed_problem.L1_error[i] << std::endl;
                
                for(unsigned int i=0; i<ed_problem.L2_error.size(); ++i)
                    fp << ed_problem.L2_error[i] << std::endl;
                
                fp.close();
            }
        
        }
        else if (dim==2)
        {
            ContinuousGalerkin::ElasticProblem<2> ed_problem(p, false, timing_stream);
            
            switch(scheme)
            {
                case 0:
                    ed_problem.run_AvgAccel(time_integrator[scheme], h,h);
                    break;
                case 1:
                    ed_problem.run_BackwardEuler(time_integrator[scheme], h,h);
                    break;
                case 2:
                    ed_problem.run_Bathe(time_integrator[scheme], h,h);
                    break;
                case 3:
                    ed_problem.run_CDF(time_integrator[scheme], h,h);
                    break;
                case 4:
                    ed_problem.run_ExpGenAlpha(time_integrator[scheme], h,h);
                    break;
                case 5:
                    ed_problem.run_ForwardEuler(time_integrator[scheme], h,h);
                    break;
                case 6:
                    ed_problem.run_GenAlpha(time_integrator[scheme], h,h);
                    break;
                case 7:
                    ed_problem.run_HHT(time_integrator[scheme], h,h);
                    break;
                case 8:
                    ed_problem.run_LinAccel(time_integrator[scheme], h,h);
                    break;
                case 9:
                    ed_problem.run_WBZ(time_integrator[scheme], h,h);
                    break;
                default:
                    break;
                    
            }
            
            n_cells = ed_problem.n_cells;
            n_dofs = ed_problem.n_dofs;
            
            timing_stream.close();
            
            {
                    // Dump to a file:
                    // polynomial order
                    // nx
                    // cells
                    // dofs
                    // L1 errors
                    // L2 errors
                std::string fileName = "./" + time_integrator[scheme] + "Errors_d"+sd+"1_p"
                + sp + "_h" + sh + ".dat";
                std::fstream fp;
                fp.open(fileName.c_str(), std::ios::out);
                fp << p << std::endl;
                fp << h << std::endl;
                fp << n_cells << std::endl;
                fp << n_dofs << std::endl;
                
                fp.precision(16);
                
                for(unsigned int i=0; i<ed_problem.L1_error.size(); ++i)
                    fp << ed_problem.L1_error[i] << std::endl;
                
                for(unsigned int i=0; i<ed_problem.L2_error.size(); ++i)
                    fp << ed_problem.L2_error[i] << std::endl;
                
                fp.close();
            }
        }
        else
        {
            std::cout<<"\n dim=3 not tested!"<<std::endl;
            exit(1);
        }
        
//    timing_stream.close();
//            
//        {
//        // Dump to a file:
//        // polynomial order
//        // nx
//        // cells
//        // dofs
//        // L1 errors
//        // L2 errors
//        std::string fileName = "./" + time_integrator[scheme] + "Errors_d1_p"
//                        + sp + "_h" + sh + ".dat";
//        std::fstream fp;
//        fp.open(fileName.c_str(), std::ios::out);
//        fp << p << std::endl;
//        fp << h << std::endl;
//        fp << n_cells << std::endl;
//        fp << n_dofs << std::endl;
//        
//        fp.precision(16);
//        
//        for(unsigned int i=0; i<ed_problem.L1_error.size(); ++i)
//            fp << ed_problem.L1_error[i] << std::endl;
//        
//        for(unsigned int i=0; i<ed_problem.L2_error.size(); ++i)
//            fp << ed_problem.L2_error[i] << std::endl;
//        
//        fp.close();
//        }

    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    
    return 0;
}//main




