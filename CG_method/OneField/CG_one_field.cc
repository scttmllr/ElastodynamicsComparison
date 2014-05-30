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
	
	//dealii::Vector<double> coefficients;
	//ConstraintMatrix     hanging_node_constraints;
	
	CompressedSparsityPattern      sparsity_pattern;
	
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
    Vector<double> rhs;
    Vector<double> solution, old_solution, mid_solution;
	
	// Also, keep track of the current time and the time spent evaluating
    // certain functions
    Time                             time;
    TimerOutput                      timer_output;
    
    // Info for time stepping:
    double current_time, final_time, dt;
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
	VectorTools::interpolate_boundary_values (MappingQ1<dim>(),
											dof_handler,
											ExactSolution<dim>(dim, current_time),
											constraints);
    
	constraints.close ();
	
	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
	sparsity_pattern.copy_from(c_sparsity);
	
	
	system_matrix.reinit (sparsity_pattern);
    
    rhs.reinit(dof_handler.n_dofs());
    
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    mid_solution.reinit(dof_handler.n_dofs());
}//setup_system


template <int dim>
void ElasticProblem<dim>::assemble_implicit_system ()
{
    // Zero matrices and vectors
    system_matrix = 0.;
    consistent_mass_matrix = 0.;
    stiffness_matrix = 0.;
    inverse_system_matrix = 0.;

    rhs = 0.;

    QGauss<dim>  quadrature_formula(std::ceil(((2.0*fe.degree) +1)/2));

    FEValues<dim> fe_values (fe, quadrature_formula,
                         update_values   | 
                         update_quadrature_points | 
                         update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    // Now we can begin with the loop
    // over all cells:
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();

    double rho, lambda, mu;

    for (; cell!=endc; ++cell)
 	{
 		cell_matrix = 0;
 		
 		get_material_paras(cell->material_id(), rho, lambda, mu);
 		
 		fe_values.reinit (cell);
 		
 		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
 		{
 			for (unsigned int i=0; i<dofs_per_cell; ++i)
 			{
 				for (unsigned int j=0; j<dofs_per_cell; ++j)
 				{
 					cell_matrix(i,j) += fe_values[disp].value(i,q_point) * 
 					fe_values[disp].value(j,q_point) *
 					fe_values.JxW(q_point);
 					
 					cell_matrix(i,j) += rho * fe_values[vel].value(i,q_point) * 
 					fe_values[vel].value(j,q_point) *
 					fe_values.JxW(q_point);
 				}//j
 			}//i
 		}//q_point
 		
 		cell->get_dof_indices (local_dof_indices);
 		for (unsigned int i=0; i<dofs_per_cell; ++i)
 		{
 			for (unsigned int j=0; j<dofs_per_cell; ++j)
 				mass_matrix.add (local_dof_indices[i],
 								 local_dof_indices[j],
 								 cell_matrix(i,j));
 		}//i
 		
 	}//cell
 	
 	Vector<double> ones;
 	ones.reinit(coefficients.size());
 	coefficients = 0.;
 	ones = 1.;
 	
 	// Need an exact solution function!
 		//boundary_values.clear();
 	VectorTools::interpolate_boundary_values (dof_handler,
 											  0,
 											  ExactSolution<dim>(2*dim, td.current_time()),
 											  boundary_values);
 	MatrixTools::apply_boundary_values (boundary_values,
 										mass_matrix,
 										coefficients,
 										ones, 
 										false);
 	
 	directSolver.initialize (mass_matrix);
 }//assemble_mass_matrix


template <int dim>
void ElasticProblem<dim>::solve (Vector<double> &soln, Vector<double> &rhs)
{
    SparseDirectUMFPACK directSolver;
	directSolver.initialize (system_matrix);
	directSolver.vmult (solution, rhs);

    constraints.distribute (solution);
}//solve

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
	
	data_out.add_data_vector (time, solution_names,
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
	Vector<double> local_errors (triangulation.n_active_cells());
	
	for(int i=0; i<2*dim; ++i){
	
	ComponentSelectFunction<dim> mask(i, 2*dim);
		
	local_errors = 0.0;
	VectorTools::integrate_difference(dof_handler, 
									td.access_current_solution(),
									ExactSolution<dim>(2*dim, td.current_time()),
									local_errors,
									QGauss<dim>(fe.degree+2),
									VectorTools::L2_norm,
									&mask);
	
	L2_error[i] = local_errors.l2_norm();
	
	local_errors = 0.0;
	VectorTools::integrate_difference(dof_handler, 
									td.access_current_solution(),
									ExactSolution<dim>(2*dim, td.current_time()),
									local_errors,
									QGauss<dim>(fe.degree+2),
									VectorTools::L1_norm,
									&mask);
	
	L1_error[i] = local_errors.l1_norm();
	}
	
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
	
		// Project IC's here.  All zero for our problem.
// 	if(dim==2)         
// 	{
// 		old_solution = 0.0;     
// 	}
// 	else
// 	{
// 		FunctionParser<dim> initial_conditions(2*dim);
// 		std::vector<std::string> expressions (2*dim, "0.0");
// 		expressions[dim] = "1.0"; // set velocity in x-direction
// 		
// 		initial_conditions.initialize (FunctionParser<dim>::default_variable_names(),
// 									   expressions,
// 									   std::map<std::string, double>());
// 		
// 		VectorTools::interpolate(dof_handler, initial_conditions, old_solution);
// 	}
	
// Time discretization runs it:
		
	td.set_time_scheme(time_integrator);
	td.reinit(dof_handler.n_dofs(), 0.0);
	
	VectorTools::interpolate(dof_handler, 
				ExactSolution<dim>(2*dim, td.current_time()),
				td.access_current_solution());
				
	if(time_integrator == "Bathe"){
		VectorTools::interpolate(dof_handler, 
				ExactSolutionTimeDerivative<dim>(2*dim, td.current_time()),
				td.access_solution_derivative());
	}
				
	//output_results(0, time_integrator);
	
    double final_time = dim == 1 ? 1.0 : 0.5;//0.609449;
    
    //final_time = 0.125;
    // should base this on cfl?
	double delta_t = 0.001;//
	//0.000005;//0.01*(1./double(nx));
	
	if(time_integrator=="Bathe")
		delta_t = 0.0001;
	else
		delta_t = 1.e-5;

	td.set_final_time(final_time);
	td.set_delta_t(delta_t);
	
	while (!td.finalized() )
    {
//    std::cout<<"\nBefore advance"<<std::endl;
		td.advance();
		
        //if(td.time_step_num() % 1000 == 0)
        //    output_results (td.time_step_num());
		
		//std::cout<<"\nl2 norm of solution = "<<td.access_current_solution().l2_norm()<<std::endl;
	}//while
	
	output_results (td.time_step_num(), time_integrator);
	
}
    
}//namespace

/***********************************
 * MAIN
 ***********************************/
int main ()
{
    try
    {
     int np=3, nh=8, nt=3;
     
     int nx[8] = {1, 2, 4, 8, 16, 32, 64, 128};
     int p[5] = {1, 2, 5};
     //int p[5] = {1, 2, 3, 4, 5};
     
     std::string snx[8] = {"1", "2", "4", "8", "16", "32", "64", "128"};
     std::string sp[5] = {"1", "2", "5"};
     //std::string sp[5] = {"1", "2", "3", "4", "5"};
     
     std::vector<std::string> timeint(3);
     timeint[0] = "RK1";
     timeint[1] = "SSP_5_4";
     timeint[2] = "Bathe";
     
// 	for(int j=0; j<np; ++j)     
// 		for(int k=0; k<nh; ++k)
//    			for(int i=0; i<nt; ++i)
//         {
// 	  	timer.start();
//         ContinuousGalerkin::ElasticProblem<1> ed_problem(p[j]);
//         ed_problem.run (timeint[i], nx[k]);
//         timer.stop();
//         ed_problem.compute_errors();
//         std::cout << "\nElapsed CPU time: " << timer() << " seconds.";
// 		std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";
// 		
// 		std::string fileName = "./output_d1/" + timeint[i] + "_p" + sp[j] + "_h" + snx[k] + ".dat";
// 		std::fstream fp;
// 		fp.open(fileName.c_str(), std::ios::out);
// 		fp.precision(16);
// 		fp<<timer()<<'\n'<<timer.wall_time()<<'\n';
// 		for(int i=0; i<2; ++i)
// 			fp<<std::setprecision(16)<<ed_problem.L1_error[i]<<'\n';
// 		for(int i=0; i<2; ++i)
// 			fp<<std::setprecision(16)<<ed_problem.L2_error[i]<<'\n';
// 		fp.close();
// 		
// 		timer.reset();
// 		}
		
	for(int j=0; j<np; ++j)     
		for(int k=0; k<nh; ++k)
   			for(int i=0; i<nt; ++i)
        {
	  	timer.start();
        ContinuousGalerkin::ElasticProblem<2> ed_problem(p[j]);
        ed_problem.run (timeint[i], nx[k], nx[k]);
        timer.stop();
        ed_problem.compute_errors();
        std::cout << "\nElapsed CPU time: " << timer() << " seconds.";
		std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";
		
		std::string fileName = "./output_d2/" + timeint[i] + "_p" + sp[j] + "_h" + snx[k] + ".dat";
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




