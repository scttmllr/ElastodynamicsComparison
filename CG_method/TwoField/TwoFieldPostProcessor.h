/******************************************** 
     * POSTPROCESSOR
     ********************************************/
template<int dim>
class ElasticPostprocessor: public DataPostprocessor<dim>
{
public:
	ElasticPostprocessor ();
	
	virtual
	void
	compute_derived_quantities_vector (const std::vector<Vector<double> > &uh,
									   const std::vector<std::vector<Tensor<1,dim> > > &duh,
									   const std::vector<std::vector<Tensor<2,dim> > > &dduh,
									   const std::vector<Point<dim> >                  &normals,
									   const std::vector<Point<dim> >                  &evaluation_points,
									   std::vector<Vector<double> >                    &computed_quantities) const;
	
	virtual std::vector<std::string> get_names () const;
	
	virtual
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	get_data_component_interpretation () const;
	
	virtual UpdateFlags get_needed_update_flags () const;
	
	virtual unsigned int n_output_variables() const;
};

template <int dim>
ElasticPostprocessor<dim>::ElasticPostprocessor()
{}

template <int dim>
void
ElasticPostprocessor<dim>::
compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
								   const std::vector<std::vector<Tensor<1,dim> > > &duh,
								   const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
								   const std::vector<Point<dim> >                  &/*normals*/,
								   const std::vector<Point<dim> >                  &evaluation_points,
								   std::vector<Vector<double> >                    &computed_quantities) const
{
	const unsigned int n_quadrature_points = evaluation_points.size();
	
	Assert (computed_quantities.size() == n_quadrature_points,
			ExcInternalError());
	
	double rho, lambda, mu;
	
	if(dim==2)
	{
		rho = 1.;
		lambda = 2.;
		mu = 0.5;
	}
	else
	{
		rho = 1.;
		lambda = 7.92e+06;
		mu = 9.69e+06;
	}
	
	Tensor<2,dim> S, E;
	
	Tensor<2,dim> Id;
	Id = 0.0;
	for(int d=0; d<dim; ++d)
		Id[d][d] = 1;
	
	double trE;
	
	Tensor<2,dim> grad_u;
	
	for (unsigned int q=0; q<n_quadrature_points; ++q)
	{
		for(int i=0; i<dim; ++i)
			for(int j=0; j<dim; ++j)
				grad_u[i][j] = duh[q][i][j];
		
		E = 0.5*(grad_u + transpose(grad_u));
		
		trE = 0.0;
		for(int d=0; d<dim; ++d)
			trE += E[d][d];
		
		S = Id;
		S *= (lambda*trE);
		S += (2.0*mu)*E;
		
		if(dim==1)
			computed_quantities[q](0) = S[0][0];
		else if(dim==2)
		{
			computed_quantities[q](0) = S[0][0];
			computed_quantities[q](1) = S[1][0];
			computed_quantities[q](2) = S[1][1];
		}
		else
		{
			computed_quantities[q](0) = S[0][0];
			computed_quantities[q](1) = S[0][1];
			computed_quantities[q](2) = S[0][2];
			computed_quantities[q](3) = S[1][1];
			computed_quantities[q](4) = S[1][2];
			computed_quantities[q](5) = S[2][2];
		}
	}
	
}//compute_derived_quantities_vector

template <int dim>
std::vector<std::string>
ElasticPostprocessor<dim>::
get_names () const
{
	std::vector<std::string> names;
	
	if(dim==1)
	{
		names.push_back("S_x");
	}
	if(dim==2)
	{
		names.push_back("S_xx");
		names.push_back("S_xy");
		names.push_back("S_yy");
	}
	else
	{
		names.push_back("S_xx");
		names.push_back("S_xy");
		names.push_back("S_xz");
		names.push_back("S_yy");
		names.push_back("S_yz");
		names.push_back("S_zz");
	}
	
	return names;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
ElasticPostprocessor<dim>::
get_data_component_interpretation () const
{
	unsigned int num=3;
	
	if(dim==3)
		num=6;
	
		// We just want to output stress for now
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	interpretation (num, DataComponentInterpretation::component_is_scalar);			      
	
	return interpretation;
}



template <int dim>
UpdateFlags
ElasticPostprocessor<dim>::
get_needed_update_flags () const
{
	return update_values | update_quadrature_points | update_gradients;
}



template <int dim>
unsigned int
ElasticPostprocessor<dim>::
n_output_variables () const
{
	if(dim==2)
		return 3;
	else
		return 6;
	
	return 0;
}