// Material Parameters for this exact solution
// Global variables for ElasticProblem:
double rho = 1.;

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

double cd(int dim){if(dim==1) return 1;
else {
	double E = 1.0;
	double nu = 0.3;
//return std::sqrt((E*(1.-nu))/(rho*(1.+nu)*(1.-2.*nu)));
     double l = lambda(dim);
     double m = mu(dim);

    return std::sqrt( (l+2.*m)/rho );
}
}//cd

double A_0=1.0;
double A_1=-A_0;

double m_0 = nc::twoPi*A_0;
double m_1 = nc::twoPi*A_1;

double alpha1= nc::twoPi;
double alpha = nc::twoPi*std::sqrt(2)*cd(2);

template <int dim>
class ExactSolution : public Function<dim>
{
public:
ExactSolution (const unsigned int n_components = 1,
			   const double time = 0.) : Function<dim>(n_components, time) {}
virtual double value (const Point<dim> &p,
					  const unsigned int component = 0) const;
};

template < >
double ExactSolution<1>::value (const Point<1> &p,
								const unsigned int comp) const
{
	double t = this->get_time ();

// We should have m_i's, A_i's and \alpha1 available here as global constructs.
	if (comp == 0) // displacement
		return A_0*std::sin(m_0*p(0))*std::sin(alpha1*t);

	if (comp == 1) // velocity
		return alpha1*A_0*std::sin(m_0*p(0))*std::cos(alpha1*t);

	if (comp == 2) // strain
		return m_0*A_0*std::cos(m_0*p(0))*std::sin(alpha1*t);
		
	return -1;
}

template < >
double ExactSolution<2>::value (const Point<2> &p,
								const unsigned int comp) const
{
	double t = this->get_time ();

// We should have m_i's, A_i's and \alpha available here as global constructs.
	//  displacements
	if (comp == 0) // u_x
		return A_0*std::sin(m_0*p(0))*std::sin(m_1*p(1))*std::sin(alpha*t);

	if (comp == 1) // u_y
		return A_1*std::cos(m_0*p(0))*std::cos(m_1*p(1))*std::sin(alpha*t);

	if (comp == 2) // v_x
		return alpha*A_0*std::sin(m_0*p(0))*std::sin(m_1*p(1))*std::cos(alpha*t);

	if (comp == 3) // v_y
		return alpha*A_1*std::cos(m_0*p(0))*std::cos(m_1*p(1))*std::cos(alpha*t);
		
	if (comp == 4) //E_xx
		return m_0*A_0*std::cos(m_0*p(0))*std::sin(m_1*p(1))*std::sin(alpha*t);
		
	if (comp == 6) //E_xy
		return 0.50*(m_1*A_0*std::sin(m_0*p(0))*std::cos(m_1*p(1))*std::sin(alpha*t) -
			m_0*A_1*std::sin(m_0*p(0))*std::cos(m_1*p(1))*std::sin(alpha*t));

	if (comp == 5) // E_yy
		return -1.*m_1*A_1*std::cos(m_0*p(0))*std::sin(m_1*p(1))*std::sin(alpha*t);
		
	return -1;
}

template < >
double ExactSolution<3>::value (const Point<3> &p,
								const unsigned int comp) const
{
	std::cout<<"\nExactSolution::value not implemented for d==3!"<<std::endl;
	exit(1);
		
	return -1;
}
/**********
TIME DERIVATIVE
**********/
template <int dim>
class ExactSolutionTimeDerivative : public Function<dim>
{
public:
ExactSolutionTimeDerivative (const unsigned int n_components = 1,
			   const double time = 0.) : Function<dim>(n_components, time) {}
virtual double value (const Point<dim> &p,
					  const unsigned int component = 0) const;
};

template < >
double ExactSolutionTimeDerivative<1>::value (const Point<1> &p,
								const unsigned int comp) const
{
	double t = this->get_time ();

	if (comp == 0) // displacement
		return alpha1*A_0*std::sin(m_0*p(0))*std::cos(alpha1*t);

	if (comp == 1) // velocity
		return -1.0*alpha1*alpha1*A_0*std::sin(m_0*p(0))*std::sin(alpha1*t);

	if (comp == 2) // strain
		return alpha1*m_0*A_0*std::cos(m_0*p(0))*std::cos(alpha1*t);
		
	return -1;
}

template < >
double ExactSolutionTimeDerivative<2>::value (const Point<2> &p,
								const unsigned int comp) const
{
	double t = this->get_time ();

	if (comp == 0) // u_x
		return alpha*A_0*std::sin(m_0*p(0))*std::sin(m_1*p(1))*std::cos(alpha*t);

	if (comp == 1) // u_y
		return alpha*A_1*std::cos(m_0*p(0))*std::cos(m_1*p(1))*std::cos(alpha*t);

	if (comp == 2) // v_x
		return -1.0*alpha*alpha*A_0*std::sin(m_0*p(0))*std::sin(m_1*p(1))*std::sin(alpha*t);

	if (comp == 3) // v_y
		return -1.0*alpha*alpha*A_1*std::cos(m_0*p(0))*std::cos(m_1*p(1))*std::sin(alpha*t);
		
	if (comp == 4) //E_xx
		return alpha*m_0*A_0*std::cos(m_0*p(0))*std::sin(m_1*p(1))*std::cos(alpha*t);
		
	if (comp == 6) //E_xy
		return alpha*0.50*(m_1*A_0*std::sin(m_0*p(0))*std::cos(m_1*p(1))*std::cos(alpha*t) -
			alpha*m_0*A_1*std::sin(m_0*p(0))*std::cos(m_1*p(1))*std::cos(alpha*t));

	if (comp == 5) // E_yy
		return -1.*alpha*m_1*A_1*std::cos(m_0*p(0))*std::sin(m_1*p(1))*std::cos(alpha*t);
		
	return -1;
}

template < >
double ExactSolutionTimeDerivative<3>::value (const Point<3> &p,
								const unsigned int comp) const
{
	std::cout<<"\nExactSolution::value not implemented for d==3!"<<std::endl;
	exit(1);
		
	return -1;
}

//Instantiate the ExactSolutionClass?
