#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

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

#include "../../include/user_defined_constants.h"
#include "../../include/ExactSolutionFunction.h"


/******************************************** 
 * ELASTIC PROBLEM
 ********************************************/
template <int dim>
class ElasticProblem
{
public:
	ElasticProblem (const int poly_order, const bool lumped_mass);
	~ElasticProblem ();
	void run (std::string time_integrator, int nx, int ny=-1, int nz=-1);
	void compute_errors ();
	std::vector<double> L1_error, L2_error;
	
private:
	
	void setup_system ();
	
	void assemble_implicit_system();
	void assemble_explicit_system();
	
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
    Vector<double> solution, old_solution, old_velocity;
    
        // Temp vector for doing linear combinations of Vectors
    Vector<double> linear_combo;
	
	// Also, keep track of the current time and the time spent evaluating
    // certain functions
//    TimerOutput                      timer_output;
    
    // Info for time stepping:
    double current_time;
    unsigned int n_timesteps;
	
	// Use Q1 mapping (straight edges) for everything.
	// It is the default, so we need not explicitly use it.
//	const MappingQ1<dim> mapping;
	
	const FEValuesExtractors::Vector  disp;
};


template <int dim>
ElasticProblem<dim>::ElasticProblem (const int poly_order, const bool lumped_mass)
:
dof_handler (triangulation),
fe (FE_Q<dim>(poly_order), dim),
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
    linear_combo.reinit(dof_handler.n_dofs());
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

    constraints.distribute_local_to_global(cell_mass, local_dof_indices, stiffness_matrix);
 		
 	}//cell

 }//assemble_mass_matrix

template <int dim>
void ElasticProblem<dim>::output_results (const unsigned int cycle, std::string time_integrator) const
{
	std::string filename = "./output_d";
	filename += Utilities::int_to_string(dim,1);
	filename += "/";//solution";//+Utilities::int_to_string (cycle, 6)
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
//	Vector<double> local_errors (triangulation.n_active_cells());
//	
//	for(int i=0; i<2*dim; ++i){
//	
//	ComponentSelectFunction<dim> mask(i, 2*dim);
//		
//	local_errors = 0.0;
//	VectorTools::integrate_difference(dof_handler, 
//									td.access_current_solution(),
//									ExactSolution<dim>(2*dim, td.current_time()),
//									local_errors,
//									QGauss<dim>(fe.degree+2),
//									VectorTools::L2_norm,
//									&mask);
//	
//	L2_error[i] = local_errors.l2_norm();
//	
//	local_errors = 0.0;
//	VectorTools::integrate_difference(dof_handler, 
//									td.access_current_solution(),
//									ExactSolution<dim>(2*dim, td.current_time()),
//									local_errors,
//									QGauss<dim>(fe.degree+2),
//									VectorTools::L1_norm,
//									&mask);
//	
//	L1_error[i] = local_errors.l1_norm();
//	}
	
}//compute_errors

template <int dim>
void ElasticProblem<dim>::run (std::string time_integrator, int nx, int ny, int nz)
{
	create_grid(nx, ny, nz);
	
	std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells()
            << std::endl;
	
	setup_system ();
	
	std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
    
        // Set time stepping parameters:
				
	//output_results(0, time_integrator);
	
    double final_time = dim == 1 ? 1.0 : 0.5;
    
        // Mesh size:
    double h = 1./double(nx);
    
        // For a CFL of 0.5:
    double cfl = 0.50;
    double delta_t = cfl*h/cd(dim);
    double inv_dt = 1./delta_t;
    unsigned int n_timesteps = final_time / delta_t;
    
    std::cout << "\n Wave speed:      cd = " << cd(dim);
    std::cout << "\n Mesh size:       h  = " << h;
    std::cout << "\n Time step size:  dt = " << delta_t;
    std::cout << "\n Number of steps: N  = " << n_timesteps;
    
        // Now actually do the work:
    current_time = 0.;
    
        // Set Initial Conditions
	VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(dim, current_time),
                             solution);
    
    VectorTools::interpolate(dof_handler,
                             ExactSolutionTimeDerivative<dim>(dim, current_time),
                             old_velocity);
    
    for(unsigned int step=0; step<n_timesteps; ++step)
    {
        current_time += delta_t;
        
        assemble_implicit_system();
        
        system_rhs = 0.0;
        linear_combo = old_solution;
        linear_combo *= inv_dt;
        linear_combo += old_velocity;
        
        consistent_mass_matrix.vmult(system_rhs, linear_combo);
        system_rhs *= inv_dt;
        
        system_matrix = 0.;
        system_matrix.add(delta_t,consistent_mass_matrix);
        system_matrix.add(1.0,stiffness_matrix);
        
        SparseDirectUMFPACK directSolver;
        directSolver.initialize (system_matrix);
        directSolver.vmult (solution, system_rhs);
        
        constraints.distribute (solution);
    }
	
	output_results (n_timesteps, time_integrator);
	
}
    
}//namespace

/***********************************
 * MAIN
 ***********************************/
int main ()
{
    try
    {
    int np=5, nh=8;

    int nx[8] = {1, 2, 4, 8, 16, 32, 64, 128};
            //int p[5] = {1, 2, 5};
    int p[5] = {1, 2, 3, 4, 5};

    std::string snx[8] = {"1", "2", "4", "8", "16", "32", "64", "128"};
        //std::string sp[5] = {"1", "2", "5"};
    std::string sp[5] = {"1", "2", "3", "4", "5"};
        
    std::string time_integrator = "BackwardEuler";
        
    dealii::Timer timer;
     
 	for(int j=0; j<np; ++j)     
        for(int k=0; k<nh; ++k)
        {
            timer.start();
            ContinuousGalerkin::ElasticProblem<1> ed_problem(p[j], false);
            ed_problem.run (time_integrator, nx[k]);
            timer.stop();
            ed_problem.compute_errors();
            std::cout << "\nElapsed CPU time: " << timer() << " seconds.";
            std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";

            std::string fileName = "./output_d1/" + time_integrator + "_p" + sp[j] + "_h" + snx[k] + ".dat";
            std::fstream fp;
            fp.open(fileName.c_str(), std::ios::out);
            fp.precision(16);
            fp<<timer()<<'\n'<<timer.wall_time()<<'\n';
            for(int i=0; i<2; ++i)
            fp<<std::setprecision(16)<<ed_problem.L1_error[i]<<'\n';
            for(int i=0; i<2; ++i)
            fp<<std::setprecision(16)<<ed_problem.L2_error[i]<<'\n';
            fp.close();

            timer.reset();
        }
		
            // dim = 2
//	for(int j=0; j<np; ++j)     
//		for(int k=0; k<nh; ++k)
//   			for(int i=0; i<nt; ++i)
//        {
//	  	timer.start();
//        ContinuousGalerkin::ElasticProblem<2> ed_problem(p[j]);
//        ed_problem.run (timeint[i], nx[k], nx[k]);
//        timer.stop();
//        ed_problem.compute_errors();
//        std::cout << "\nElapsed CPU time: " << timer() << " seconds.";
//		std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";
//		
//		std::string fileName = "./output_d2/" + timeint[i] + "_p" + sp[j] + "_h" + snx[k] + ".dat";
//		std::fstream fp;
//		fp.open(fileName.c_str(), std::ios::out);
//		fp.precision(16);
//		fp<<timer()<<'\n'<<timer.wall_time()<<'\n';
//		for(int i=0; i<2; ++i)
//			fp<<std::setprecision(16)<<ed_problem.L1_error[i]<<'\n';
//		for(int i=0; i<2; ++i)
//			fp<<std::setprecision(16)<<ed_problem.L2_error[i]<<'\n';
//		fp.close();
//		
//		timer.reset();
//		}

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




