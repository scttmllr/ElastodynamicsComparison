/******************************************** 
 * RIGHT HAND SIDE
 ********************************************/
template <int dim>
class RightHandSide :  public Function<dim>
{
public:
	RightHandSide ();
	
	
	virtual void vector_value (const Point<dim> &p,
							   Vector<double>   &values) const;
	
	virtual void vector_value_list (const std::vector<Point<dim> > &points,
									std::vector<Vector<double> >   &value_list) const;
};

template <int dim>
RightHandSide<dim>::RightHandSide ()
:
Function<dim> (dim)
{}


template <int dim>
inline
void RightHandSide<dim>::vector_value (const Point<dim> &p,
									   Vector<double>   &values) const
{
	Assert (values.size() == dim,
			ExcDimensionMismatch (values.size(), dim));
	
	unused(p);
	
	values = 0.00;
	
	if(dim==2)
		values(1) = -9.81; // gravity, y-axis
	
}//RHS::vector_value


template <int dim>
void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
											std::vector<Vector<double> >   &value_list) const
{
	Assert (value_list.size() == points.size(),
			ExcDimensionMismatch (value_list.size(), points.size()));
	
	const unsigned int n_points = points.size();
	
	for (unsigned int p=0; p<n_points; ++p)
		RightHandSide<dim>::vector_value (points[p],
										  value_list[p]);
}//RHS::vector_value_list
