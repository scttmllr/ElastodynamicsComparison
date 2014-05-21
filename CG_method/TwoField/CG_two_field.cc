#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_in.h>
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
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

    // MeshWorker specific
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>


#include <fstream>
#include <iostream>


namespace ContinuousGalerkin
{
using namespace dealii;

	//! Template function that can be used to suppress compiler warnings
template<class T> void unused( const T& ) { }

// Global variables for ElasticProblem:
double rho = 1.;
//double E = 1.0;
//double nu = 0.3;

double cd(int dim){if(dim==1) return 1;
else {
	double E = 1.0;
	double nu = 0.3;
return std::sqrt((E*(1.-nu))/(rho*(1.+nu)*(1.-2.*nu)));
}
}//cd

double lambda(int dim)
{
	if (dim==1)
		return 0.5;
	
	return 0.576923;
}//lambda

double mu(int dim)
{
	if (dim==1)
		return 0.25;
	
	return 0.384615;
}//lambda

//double lambda = 0.5;//0.576923;
//(E*nu)/((1.+nu)*(1.-2.*nu));
//double mu = 0.25;//0.384615;
//E/(2.*(1.+nu));

#include "../../include/user_defined_constants.h"
#include "../../include/RightHandSide.h"
//#include "../../include/gTime.h"
#include "../../include/ExactSolutionFunction.h"
#include "../../include/CG_time_discretization.h"

#include "TwoFieldPostProcessor.h"


/******************************************** 
 * ELASTIC PROBLEM
 ********************************************/
template <int dim>
class ElasticProblem
{
public:
	ElasticProblem (int poly_order);
	~ElasticProblem ();
	void run (std::string time_integrator, int nx, int ny=-1, int nz=-1);
	void compute_errors ();
	std::vector<double> L1_error, L2_error;
	
private:
		// MESHWORKER
	typedef MeshWorker::DoFInfo<dim> DoFInfo;
	typedef MeshWorker::IntegrationInfo<dim> CellInfo;
	
	void setup_system ();
	void assemble_system (Vector<double> &dest, Vector<double>& soln, bool assemble_jacobian,
		double mass_coeff=1.0);
	void assemble_system_Bathe (Vector<double> &dest, Vector<double>& soln, bool assemble_jacobian,
		double mass_coeff=1.0);
	void integrate_cell_term (DoFInfo& dinfo, CellInfo& info);
	void integrate_cell_term_Bathe (DoFInfo& dinfo, CellInfo& info);
	
	//void assemble_mass_matrix ();
	//void solve (Vector<double> &soln, Vector<double> &rhs);
	void refine_grid ();
	void output_results (const unsigned int cycle, std::string time_integrator) const;
	
	void create_grid (int nx, int ny=-1, int nz=-1);
	void assign_material_and_boundary_ids ();
	
	Triangulation<dim>   triangulation;
	DoFHandler<dim>      dof_handler;
	
	FESystem<dim>        fe;
	
	TimeDiscretization<dim> td;
	
	//dealii::Vector<double> coefficients;
	//ConstraintMatrix     hanging_node_constraints;
	
	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;
	
	std::map<unsigned int,double> boundary_values;
	
	const MappingQ1<dim> mapping;
	
	const FEValuesExtractors::Vector  disp, vel;
	
	bool assemble_jacobian;
	
	double mass_coeff;
    
    Vector<double> update_vector;
};


template <int dim>
ElasticProblem<dim>::ElasticProblem (int poly_order)
:
L1_error(2*dim, 0.0),
L2_error(2*dim, 0.0),
dof_handler (triangulation),
fe (FE_Q<dim>(poly_order), 2*dim),
td(
    std_cxx1x::bind(&ElasticProblem<dim>::assemble_system, 
                       this, std_cxx1x::_1, std_cxx1x::_2, std_cxx1x::_3, std_cxx1x::_4),
    std_cxx1x::bind(&ElasticProblem<dim>::assemble_system_Bathe, 
                       this, std_cxx1x::_1, std_cxx1x::_2, std_cxx1x::_3, std_cxx1x::_4),
    dof_handler
   ),
disp(0),
vel(dim)
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
	// hanging_node_constraints.clear ();
// 	DoFTools::make_hanging_node_constraints (dof_handler,
// 											 hanging_node_constraints);
// 	hanging_node_constraints.close ();
	sparsity_pattern.reinit (dof_handler.n_dofs(),
							 dof_handler.n_dofs(),
							 dof_handler.max_couplings_between_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
	
//	hanging_node_constraints.condense (sparsity_pattern);
	
	sparsity_pattern.compress();
	
	system_matrix.reinit (sparsity_pattern);
    
    update_vector.reinit(dof_handler.n_dofs());
}


template <int dim>
void ElasticProblem<dim>::assemble_system (Vector<double> &soln, Vector<double>& residual, bool assemble_jacobian,
			double mass_coeff)
{
//	std::cout<<"\nEntering assemble_system"<<std::endl;
	system_matrix = 0.0;
    update_vector = 0.0;
	
	this->assemble_jacobian = assemble_jacobian;
	this->mass_coeff = mass_coeff;
	const unsigned int n_gauss_points = std::ceil(((2.0*fe.degree) +1)/2);
	
	MeshWorker::IntegrationInfoBox<dim> info_box;
	
	info_box.initialize_gauss_quadrature(n_gauss_points,
										 n_gauss_points,
										 n_gauss_points);
	
	info_box.initialize_update_flags();
	UpdateFlags update_flags = update_quadrature_points |
	update_values |
	update_gradients;
	
	info_box.add_update_flags(update_flags, true, true, true, true);
	
	NamedData<Vector<double>* > solution_data;
	
	Vector<double>* u = &soln;
	
	solution_data.add(u, "solution");
	
	info_box.cell_selector.add("solution", true, true, false); 
	info_box.boundary_selector.add("solution", false, false, false);// Don't need face/boundary at all!
	info_box.face_selector.add("solution", false, false, false);// Don't need face/boundary at all!
	
	info_box.initialize(fe, mapping, solution_data);
	
	MeshWorker::DoFInfo<dim> dof_info(dof_handler);
	
// 	NamedData<Vector<double>* > data;
// 	Vector<double>* rhs = &dest;
// 	data.add(rhs, "RHS");
	
//	MeshWorker::Assembler::ResidualSimple<Vector<double> > assembler;
//	assembler.initialize(data);
	
	MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> > assembler;
    assembler.initialize(system_matrix, residual);
	
//std::cout<<"\nBefore meshworkder"<<std::endl;	
	MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
	(dof_handler.begin_active(), dof_handler.end(),
	 dof_info, info_box,
	 std_cxx1x::bind(&ElasticProblem<dim>::integrate_cell_term, 
					this, std_cxx1x::_1, std_cxx1x::_2),
	 NULL,
	 NULL,
	 assembler,true,true);
	 
	 // Now we actually solve and apply the BC's
// 	VectorTools::interpolate_boundary_values (dof_handler,
// 											  0,
// 											  ZeroFunction<dim>(2*dim),//ExactSolution<dim>(2*dim, td.current_time()),
// 											  boundary_values);
//     
//     if(dim==1)
//         VectorTools::interpolate_boundary_values (dof_handler,
//                                                   1,
//                                                   ZeroFunction<dim>(2*dim),//ExactSolution<dim>(2*dim, td.current_time()),
//                                                   boundary_values);
// 
// 	//ExactSolutionTimeDerivative<dim>(2*dim, td.current_time())									
// 	MatrixTools::apply_boundary_values (boundary_values,
// 		system_matrix,
// 		update_vector,
// 		residual, 
// 		false);
		

	SparseDirectUMFPACK directSolver;
	directSolver.initialize (system_matrix);
	
	directSolver.vmult (update_vector, residual);
//std::cout<<"\nAfter solve"<<std::endl;
	
//    for(unsigned int i=0; i<dof_handler.n_dofs(); ++i)
//        soln[i] = -1.0*update_vector[i];

//    update_vector *= -1.0;
//    soln = update_vector;
    residual = update_vector;
	
}//assemble_system

template <int dim>
void ElasticProblem<dim>::assemble_system_Bathe (Vector<double> &soln, Vector<double>& residual, bool assemble_jacobian,
			double mass_coeff)
{
//	std::cout<<"\nEntering assemble_system"<<std::endl;
	system_matrix = 0.0;
    update_vector = 0.0;
	
	this->assemble_jacobian = assemble_jacobian;
	this->mass_coeff = mass_coeff;
	const unsigned int n_gauss_points = std::ceil(((2.0*fe.degree) +1)/2);
	
	MeshWorker::IntegrationInfoBox<dim> info_box;
	
	info_box.initialize_gauss_quadrature(n_gauss_points,
										 n_gauss_points,
										 n_gauss_points);
	
	info_box.initialize_update_flags();
	UpdateFlags update_flags = update_quadrature_points |
	update_values |
	update_gradients;
	
	info_box.add_update_flags(update_flags, true, true, true, true);
	
	NamedData<Vector<double>* > solution_data;
	
	Vector<double>* u = &soln;
	
	solution_data.add(u, "solution");
	
	info_box.cell_selector.add("solution", true, true, false); 
	info_box.boundary_selector.add("solution", false, false, false);// Don't need face/boundary at all!
	info_box.face_selector.add("solution", false, false, false);// Don't need face/boundary at all!
	
	info_box.initialize(fe, mapping, solution_data);
	
	MeshWorker::DoFInfo<dim> dof_info(dof_handler);
	
// 	NamedData<Vector<double>* > data;
// 	Vector<double>* rhs = &dest;
// 	data.add(rhs, "RHS");
	
//	MeshWorker::Assembler::ResidualSimple<Vector<double> > assembler;
//	assembler.initialize(data);
	
	MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> > assembler;
    assembler.initialize(system_matrix, residual);
	
//std::cout<<"\nBefore meshworkder"<<std::endl;	
	MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
	(dof_handler.begin_active(), dof_handler.end(),
	 dof_info, info_box,
	 std_cxx1x::bind(&ElasticProblem<dim>::integrate_cell_term_Bathe, 
					this, std_cxx1x::_1, std_cxx1x::_2),
	 NULL,
	 NULL,
	 assembler,true,true);
	 
	 // Now we actually solve and apply the BC's
	VectorTools::interpolate_boundary_values (dof_handler,
											  0,
											  ExactSolution<dim>(2*dim, td.current_time()),
											  boundary_values);
    
    if(dim==1)
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  ExactSolution<dim>(2*dim, td.current_time()),
                                                  boundary_values);

	//ExactSolutionTimeDerivative<dim>(2*dim, td.current_time())									
	MatrixTools::apply_boundary_values (boundary_values,
		system_matrix,
		update_vector,
		residual, 
		false);
		

	SparseDirectUMFPACK directSolver;
	directSolver.initialize (system_matrix);
	
	directSolver.vmult (update_vector, residual);

    residual = update_vector;
	
}//assemble_system_Bathe


template <int dim>
void ElasticProblem<dim>::integrate_cell_term (DoFInfo& dinfo, CellInfo& info)
{
	const FEValuesBase<dim>& fe_v = info.fe_values(); 
	Vector<double>&       cell_rhs = dinfo.vector(0).block(0);
	
	const unsigned int   dofs_per_cell = fe_v.dofs_per_cell;
	
	const unsigned int   n_q_points    = fe_v.n_quadrature_points;
	
	FullMatrix<double>& cell_matrix = dinfo.matrix(0).matrix;
	
		//RightHandSide<dim>      right_hand_side;
		// std::vector<Vector<double> > rhs_values (n_q_points,
		//                                              Vector<double>(dim));
	
//	Tensor<1,dim> rhs_value;
//	rhs_value = 0.0;
	
	Tensor<2,dim> S, E, dS, dE;
	
	Tensor<2,dim> Id;
	Id = 0.0;
	for(int d=0; d<dim; ++d)
		Id[d][d] = 1;
	
	double trE, d_trE;
    Tensor<2,dim> w_i, w_j;
	
	
	std::vector<Tensor<1,dim> > u(n_q_points);
	std::vector<Tensor<1,dim> > v(n_q_points);
	std::vector<Tensor<2,dim> > grad_u(n_q_points);
	
	int first_u_comp = 0;
	int first_v_comp = dim;
	
	const std::vector<std::vector<double> > &values = info.values[0];
	const std::vector<std::vector<Tensor<1,dim> > > &grads = info.gradients[0];
	
		// Copy values into my containers
	for(unsigned int i=0; i<n_q_points; ++i)
	{
		for (unsigned int d=0; d<dim; ++d)
		{
			u[i][d] = values[first_u_comp+d][i];
			v[i][d] = values[first_v_comp+d][i];
			
			for (unsigned int q=0; q<dim; ++q)
			{
				grad_u[i][d][q] = grads[first_u_comp+d][i][q];
			}//q
			
		}//d
	}//i
	
		// Current all paras are the same...TODO FIX HACK
		//get_material_paras(cell->material_id(), rho, lambda, mu);
	
		// 	right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
		// 									   rhs_values);
		// 
	
	for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	{
		
			//F = Id + grad_u[q_point];
		E = 0.5 * (grad_u[q_point] 
				   + transpose(grad_u[q_point]) 
				   //	+ (transpose(grad_u[q_point])*grad_u[q_point])
				   );
		trE = 0.0;
		for(int d=0; d<dim; ++d)
			trE += E[d][d];
		
		S = Id;
		S *= (lambda(dim)*trE);
		S += (2.0*mu(dim))*E;
		
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
            w_i = fe_v[vel].gradient(i,q_point);
            
				//Assemble the residual
				//gravity
//			cell_rhs(i) += (fe_v[vel].value(i,q_point) *
//							rhs_value) *
//                            fe_v.JxW(q_point);
			
			cell_rhs(i) -= scalar_product(
										  w_i,
										  S )
                            * fe_v.JxW(q_point);
			
			cell_rhs(i) += v[q_point]
                            * fe_v[disp].value(i,q_point)
                            * fe_v.JxW(q_point);
			
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				{
						// "mass" matrix terms
					cell_matrix(i,j) += mass_coeff*fe_v[disp].value(i,q_point) * 
                                        fe_v[disp].value(j,q_point) *
                                        fe_v.JxW(q_point);
					
					cell_matrix(i,j) += mass_coeff*rho * fe_v[vel].value(i,q_point) * 
                                        fe_v[vel].value(j,q_point) *
                                        fe_v.JxW(q_point);
					
					if(assemble_jacobian) {
						// "stiffness" matrix terms
                        w_j = fe_v[disp].gradient(j,q_point);
                        
                        dE = 0.5 * (w_j + transpose(w_j) );
                        
                        d_trE = 0.;
                        for(int d=0; d<dim; ++d)
                            d_trE += dE[d][d];
                        
                        dS = Id;
                        dS *= (lambda(dim)*d_trE);
                        dS += (2.0*mu(dim))*dE;
                        
                        cell_matrix(i,j) -= scalar_product(w_i, dS) * fe_v.JxW(q_point);
                        
                        cell_matrix(i,j) += fe_v[disp].value(i,q_point) *
                                            fe_v[vel].value(j,q_point) *
                                            fe_v.JxW(q_point);
					}
				}//j
		}//i
	}//q_point
}//integrate_cell_term

template <int dim>
void ElasticProblem<dim>::integrate_cell_term_Bathe (DoFInfo& dinfo, CellInfo& info)
{
	const FEValuesBase<dim>& fe_v = info.fe_values(); 
	Vector<double>&       cell_rhs = dinfo.vector(0).block(0);
	
	const unsigned int   dofs_per_cell = fe_v.dofs_per_cell;
	
	const unsigned int   n_q_points    = fe_v.n_quadrature_points;
	
	FullMatrix<double>& cell_matrix = dinfo.matrix(0).matrix;
	
		//RightHandSide<dim>      right_hand_side;
		// std::vector<Vector<double> > rhs_values (n_q_points,
		//                                              Vector<double>(dim));
	
//	Tensor<1,dim> rhs_value;
//	rhs_value = 0.0;
	
	Tensor<2,dim> S, E, dS, dE;
	
	Tensor<2,dim> Id;
	Id = 0.0;
	for(int d=0; d<dim; ++d)
		Id[d][d] = 1;
	
	double trE, d_trE;
    Tensor<2,dim> w_i, w_j;
	
	
	std::vector<Tensor<1,dim> > u(n_q_points);
	std::vector<Tensor<1,dim> > v(n_q_points);
	
	int first_u_comp = 0;
	int first_v_comp = dim;
	
	const std::vector<std::vector<double> > &values = info.values[0];
	
		// Copy values into my containers
	for(unsigned int i=0; i<n_q_points; ++i)
	{
		for (unsigned int d=0; d<dim; ++d)
		{
			u[i][d] = values[first_u_comp+d][i];
			v[i][d] = values[first_v_comp+d][i];
		}//d
	}//i
	
		// Current all paras are the same...TODO FIX HACK
		//get_material_paras(cell->material_id(), rho, lambda, mu);
	
		// 	right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
		// 									   rhs_values);
		// 
	
	for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	{	
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
            w_i = fe_v[vel].gradient(i,q_point);
            
				//Assemble the residual
				//gravity
//			cell_rhs(i) += (fe_v[vel].value(i,q_point) *
//							rhs_value) *
//                            fe_v.JxW(q_point);
			
			cell_rhs(i) += u[q_point] * fe_v[disp].value(i,q_point) * fe_v.JxW(q_point);
			cell_rhs(i) += rho * v[q_point] * fe_v[vel].value(i,q_point) * fe_v.JxW(q_point);
			
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				{
						// "mass" matrix terms
					cell_matrix(i,j) += mass_coeff*fe_v[disp].value(i,q_point) * 
                                        fe_v[disp].value(j,q_point) *
                                        fe_v.JxW(q_point);
					
					cell_matrix(i,j) += mass_coeff*rho * fe_v[vel].value(i,q_point) * 
                                        fe_v[vel].value(j,q_point) *
                                        fe_v.JxW(q_point);
					
					if(assemble_jacobian) {
						// "stiffness" matrix terms
                        w_j = fe_v[disp].gradient(j,q_point);
                        
                        dE = 0.5 * (w_j + transpose(w_j) );
                        
                        d_trE = 0.;
                        for(int d=0; d<dim; ++d)
                            d_trE += dE[d][d];
                        
                        dS = Id;
                        dS *= (lambda(dim)*d_trE);
                        dS += (2.0*mu(dim))*dE;
                        
                        cell_matrix(i,j) += scalar_product(w_i, dS) * fe_v.JxW(q_point);
                        
                        cell_matrix(i,j) -= fe_v[disp].value(i,q_point) *
                                            fe_v[vel].value(j,q_point) *
                                            fe_v.JxW(q_point);
					}
				}//j
		}//i
	}//q_point
}//integrate_cell_term_Bathe

// template <int dim>
// void ElasticProblem<dim>::assemble_mass_matrix ()
// {
// 	QGauss<dim>  quadrature_formula(std::ceil(((2.0*fe.degree) +1)/2));
// 	
// 	FEValues<dim> fe_values (fe, quadrature_formula,
// 							 update_values   | 
// 							 update_quadrature_points | 
// 							 update_JxW_values);
// 	
// 	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
// 	const unsigned int   n_q_points    = quadrature_formula.size();
// 	
// 	FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
// 	
// 	std::vector<unsigned int> local_dof_indices (dofs_per_cell);
// 	
// 		// Now we can begin with the loop
// 		// over all cells:
// 	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
// 	endc = dof_handler.end();
// 	
// 	double rho, lambda, mu;
// 	
// 	for (; cell!=endc; ++cell)
// 	{
// 		cell_matrix = 0;
// 		
// 		get_material_paras(cell->material_id(), rho, lambda, mu);
// 		
// 		fe_values.reinit (cell);
// 		
// 		for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
// 		{
// 			for (unsigned int i=0; i<dofs_per_cell; ++i)
// 			{
// 				for (unsigned int j=0; j<dofs_per_cell; ++j)
// 				{
// 					cell_matrix(i,j) += fe_values[disp].value(i,q_point) * 
// 					fe_values[disp].value(j,q_point) *
// 					fe_values.JxW(q_point);
// 					
// 					cell_matrix(i,j) += rho * fe_values[vel].value(i,q_point) * 
// 					fe_values[vel].value(j,q_point) *
// 					fe_values.JxW(q_point);
// 				}//j
// 			}//i
// 		}//q_point
// 		
// 		cell->get_dof_indices (local_dof_indices);
// 		for (unsigned int i=0; i<dofs_per_cell; ++i)
// 		{
// 			for (unsigned int j=0; j<dofs_per_cell; ++j)
// 				mass_matrix.add (local_dof_indices[i],
// 								 local_dof_indices[j],
// 								 cell_matrix(i,j));
// 		}//i
// 		
// 	}//cell
// 	
// 	Vector<double> ones;
// 	ones.reinit(coefficients.size());
// 	coefficients = 0.;
// 	ones = 1.;
// 	
// 	// Need an exact solution function!
// 		//boundary_values.clear();
// 	VectorTools::interpolate_boundary_values (dof_handler,
// 											  0,
// 											  ExactSolution<dim>(2*dim, td.current_time()),
// 											  boundary_values);
// 	MatrixTools::apply_boundary_values (boundary_values,
// 										mass_matrix,
// 										coefficients,
// 										ones, 
// 										false);
// 	
// 	directSolver.initialize (mass_matrix);
// }//assemble_mass_matrix


// template <int dim>
// void ElasticProblem<dim>::solve (Vector<double> &soln, Vector<double> &rhs)
// {
// 		//    system_matrix *= 0.0;
// 		//    system_matrix.add(1.0, mass_matrix);
// 		//    
// 		//    Vector<double> coefficients, ones;
// 		//    coefficients = 0.;
// 		//    ones = 1.;
// 		//    
// 		//    std::map<unsigned int,double> boundary_values;
// 		//    VectorTools::interpolate_boundary_values (dof_handler,
// 		//                                              0,
// 		//                                              ZeroFunction<dim>(2*dim),
// 		//                                              boundary_values);
// 		//    MatrixTools::apply_boundary_values (boundary_values,
// 		//                                        system_matrix,
// 		//                                        coefficients,
// 		//                                        ones, 
// 		//                                        false);
// 	
// 		// 	SparseDirectUMFPACK directSolver;
// 		//    directSolver.initialize (system_matrix);
// 	
// 	directSolver.vmult (soln, rhs);
// 	
// 		//Now scale by coefficients to apply BCs
// 	std::map<unsigned int, double>::iterator bcmap=boundary_values.begin(), 
// 	bcmapend=boundary_values.end();
// 	
// 	unsigned int dof_id;
// 	double bc_value;
// 	
// 	for(; bcmap!=bcmapend; ++bcmap)
// 	{
// 		dof_id = bcmap->first;
// 		bc_value = bcmap->second;
// 		
// 		soln[dof_id] = coefficients[dof_id]*bc_value;
// 	}
// 	
// 		//    hanging_node_constraints.distribute (solution);
// }//solve


template <int dim>
void ElasticProblem<dim>::refine_grid ()
{
	triangulation.refine_global (1);
	dof_handler.distribute_dofs(fe);
}

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
	
	//ElasticPostprocessor<dim> ep;
	
	std::vector<std::string> solution_names(dim, "displacement");
	
	for(unsigned int d=0; d<dim; ++d)
		solution_names.push_back("velocity");
	
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	interpretation (2*dim, DataComponentInterpretation::component_is_part_of_vector);
	
	data_out.add_data_vector (td.access_current_solution(), solution_names, 
							DataOut<dim,DoFHandler<dim> >::type_automatic, 
							interpretation);
	//data_out.add_data_vector (td.access_current_solution(), ep);
	data_out.build_patches (fe.degree);
	data_out.write_vtu (output);
	
// 		//! Write out raw solution vector so I can restart if necessary!
// 	std::string fileName = "solnVector.dat";
// 	
// 	std::fstream fp;
// 	fp.open(fileName.c_str(), std::ios::out);
// 	
// 	if(!fp.is_open() )
// 	{
// 		std::cout << "\nCannot open file " << fileName << std::endl;
// 		exit(1);
// 	}
// 	
// 	fp.precision(16);
// 		// First thing we write is the length of the solution vector:
// 	fp << solution.size() << std::endl;
// 	for (unsigned int i=0; i<solution.size(); ++i)
// 		fp << solution(i) << std::endl;
// 	
// 	
// 		//fp << setprecision(16) << solution(i) << endl;
// 	
// 	fp.close();
	
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
            
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      subdivisions,
                                                      LowerLeft,
                                                      UpperRight,
                                                      false);
        }
    else if (dim == 2)
        {
        const Point<dim> LowerLeft (0.0, 0.0),
                        UpperRight (1.0, 1.0);
        
            // Define the subdivisions in the x1 and x2 coordinates.
        std::vector<unsigned int> subdivisions(dim);
        subdivisions[0] =   nx;
        subdivisions[1] =   ny;
        
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  subdivisions,
                                                  LowerLeft,
                                                  UpperRight,
                                                  false);
        }
	else if (dim == 3)
        {
            const Point<dim> LowerLeft (0.0, 0.0, 0.0),
            UpperRight (1.0, 1.0, 1.0);
            
                // Define the subdivisions in the x1 and x2 coordinates.
            std::vector<unsigned int> subdivisions(dim);
            subdivisions[0] =   nx;
            subdivisions[1] =   ny;
            subdivisions[2] =   nz;
            
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      subdivisions,
                                                      LowerLeft,
                                                      UpperRight,
                                                      false);
        }

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
        dealii::Timer timer;

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




