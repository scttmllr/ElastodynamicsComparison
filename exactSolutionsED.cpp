#include "SLPhysics.h"
#include "PhyElement.h"
#include "PhyAuxFunctions.h"
#include "SLAuxFunctions.h"
#include "PhyGlobal.h"
#include "PhyConfig.h"
#include "PhySubConfig.h"
#include "PhyIntCell.h"
#include "SLSubConfig.h"
#include "PhySecondaryFunctions.h"
#include "PhyPatch.h"


void SLSubConfig::setSpecifiedLoadFlags()
{
	if (descData.icbcNo == 8)
	{
		double L0 = 1.0;
		double L1 = 1.0;
		double L2 = 1.0;
		double f0 = 1.0;
		if (descData.data.size() > 0)
			f0 = descData.data(0);
		double f1 = 0.0, f2 = 0.0;
		int cdWave = 1;
		if (descData.flags.size() > 0)
			cdWave = descData.flags(0);
		int csWaveOption = 0;
		if (descData.flags.size() > 1)
			csWaveOption = descData.flags(1); 
		double A0 = 1.0, A1 = 0.0, A2 = 0.0;
		int crdSpaceDim = pConf->maxSpaceDimWithExtrusion;

		if (crdSpaceDim == 2)
		{
			f1 = 1.0;
			if (descData.data.size() > 1)
				f1 = descData.data(1);
			if (cdWave == 1)
			{
				A0 = f0, A1 = -f1;
			}
			else
			{
				A0 = f1, A1 = f0;
			}
		}
		if (crdSpaceDim == 3)
		{
			f1 = 1.0, f2 = 1.0;
			if (descData.data.size() > 1)
				f1 = descData.data(1);
			if (descData.data.size() > 2)
				f2 = descData.data(2);

			if (cdWave == 1)
			{
				A0 = f0, A1 = -f1, A2 = -f2;
			}
			else
			{
				if (csWaveOption == 0)
				{
					A0 = f1, A1 = f0;
				}
				else if (csWaveOption == 1)
				{
					A0 = f2, A2 = f0;
				}
				else if (csWaveOption == 2)
				{
					A0 = f0 * f2, A1 = -f1 * f2, A2 = f0 * f0 + f1 * f1;
				}
			}
		}
		double m0 = 2.0 * PI * f0 / L0;
		double m1 = 2.0 * PI * f1 / L1;
		double m2 = 2.0 * PI * f2 / L2;
		double sqrtM = sqrt(m0 * m0 + m1 * m1 + m2 * m2);
		PhyDescriptor desc;
		vector<int> flags;	flags.push_back(1);
		desc.setPhyDescriptor(flags);
		double cd = descData.materialProperties[desc][SL_CdMax]();
		double cs = descData.materialProperties[desc][SL_Cs]();
//		cout << "cd\t" << cd << endl;
//		cout << "cs\t" << cs << endl;

		double alpha;
		if (cdWave == 1)
			alpha = cd * sqrtM;
		else
			alpha = cs * sqrtM;

		descData.data.resize(11);
		descData.data(0) = f0;
		descData.data(1) = f1;
		descData.data(2) = f2;
		descData.data(3) = A0;
		descData.data(4) = A1;
		descData.data(5) = A2;
		descData.data(6) = m0;
		descData.data(7) = m1;
		descData.data(8) = m2;
		descData.data(9) = alpha;
		descData.data(10) = sqrtM;

		descData.flags.resize(2); 
		descData.flags(0) = cdWave;
		descData.flags(1) = csWaveOption;

//		cout << "descData.data\n" << descData.data << endl;
//		cout << "descData.flags\n" << descData.flags << endl;
//		getchar();
	}
}


bool SLPhysics::computeExactSolutionIndividualPhysicsLevelUVEB(int icbc, const PhyDescriptor& desc, ptCoords& crds, vTensord& disp, vTensord& vel, vTensord& strn, vTensord& b, int crdSpaceDim, VECTOR& X, double T, bool computeU, bool computeV, bool computeE, bool compB) const
{
	disp.set_vTensor(1, crdSpaceDim, tenFULL);
	vel.set_vTensor(1, crdSpaceDim, tenFULL);
	strn.set_vTensor(2, crdSpaceDim, tenSYM);
	disp.setValue(0.0);
	vel.setValue(0.0);
	strn.setValue(0.0);

	if (icbc == 8) // smooth exact solution for convergence studies
	{
		double f0, A0, m0, f1, A1, m1, f2, A2, m2; 
//		static int csWaveOption = phyConf->subConf[subConfigIndex]->descData.flags(1); // only applies for D = 3
		// As are:  (1, 0, 0) for 1D 
		// 2D, 3D longitudinal wave speed
		//			(f0, -f1, 0) wave speed
		// 2D, 3D shear wave speed
		//			(f1, f0, 0) (2D and 3D csWaveOption == 0)
		//			(f2, 0, f0) 3D csWaveOption == 1
		//			(f0.f2, -f1.f2, f0^2 + f1^2) 3D csWaveOption == 2 (normal on option == 0)

		//

		double alpha = phyConf->subConf[subConfigIndex]->descData.data(9);
//		static double sqrtM = phyConf->subConf[subConfigIndex]->descData.data(10);

//		static int cdWave = phyConf->subConf[subConfigIndex]->descData.flags(0); // applies for D = 2, 3
		// lengths in x0, x1, x2 directions
//		static double L0 = 1.0;
//		static double L1 = 1.0;
//		static double L2 = 1.0;
//		static double m0 = 2.0 * PI * f0 / L0;
//		static double m1 = 2.0 * PI * f1 / L1;
//		static double m2 = 2.0 * PI * f2 / L2;
//		static double sqrtM = sqrt(m0 * m0 + (int)(crdSpaceDim > 1) * m1 * m1 + (int)(crdSpaceDim > 2) * m2 * m2);

//		static double cd = 	getCd(desc, crds);
//		static double cs = 	getCs(desc, crds);
//		static double alpha = cdWave * cd * sqrtM + (1.0 - cdWave) * cs * sqrtM;

		if (crdSpaceDim == 1)
		{
			f0 = phyConf->subConf[subConfigIndex]->descData.data(0);
			A0 = phyConf->subConf[subConfigIndex]->descData.data(3);
			m0 = phyConf->subConf[subConfigIndex]->descData.data(6);
			double x0 = X(0), mx0 = x0 * m0, c0 = cos(mx0), s0 = sin(mx0);
			double alphaT = alpha * T, st = sin(alphaT);
			double tmp0 = A0 * s0;
			if (computeU == true)
				disp(0) = tmp0 * st;
			if (computeV == true)
			{
				double ctAlpha = cos(alphaT) * alpha;
				vel(0) = tmp0 * ctAlpha;
			}
			if (computeE == true)
				strn(0, 0) =  m0 * A0 * c0 * st;

//			disp(0) = x0 * x0 + x0 * T + T * T + pow(x0 + T, 3);
//			vel(0) = x0 + 2.0 * T + 3 * pow(x0 + T, 2);
//			strn(0, 0) = 2.0 * x0 + T + 3 * pow(x0 + T, 2);

			db << "exct xt \t" << x0 << '\t' << T;
			if (computeU == true)
				db << " u " << disp(0);
			if (computeV == true)
				db << " v " << vel(0);
			if (computeE == true)
				db << " E " << strn(0, 0);
			db << '\n';
			return false;
		}
		if (crdSpaceDim == 2)
		{
			f0 = phyConf->subConf[subConfigIndex]->descData.data(0);
			A0 = phyConf->subConf[subConfigIndex]->descData.data(3);
			m0 = phyConf->subConf[subConfigIndex]->descData.data(6);

			f1 = phyConf->subConf[subConfigIndex]->descData.data(1);
			A1 = phyConf->subConf[subConfigIndex]->descData.data(4);
			m1 = phyConf->subConf[subConfigIndex]->descData.data(7);

			double x0 = X(0), mx0 = x0 * m0, c0 = cos(mx0), s0 = sin(mx0);
			double x1 = X(1), mx1 = x1 * m1, c1 = cos(mx1), s1 = sin(mx1);
			double alphaT = alpha * T, ctAlpha = cos(alphaT) * alpha, st = sin(alphaT);
			double tmp0 = A0 * s0 * s1, tmp1 = A1 * c0 * c1;
			if (computeU == true)
			{
				disp(0) = tmp0 * st;
				disp(1) = tmp1 * st;
			}
			if (computeV == true)
			{
				vel(0) = tmp0 * ctAlpha;
				vel(1) = tmp1 * ctAlpha;
			}
			if (computeE == true)
			{
				double c0s1st = c0 * s1 * st;
				strn(0, 0) =  A0 * m0 * c0s1st;
				strn(1, 1) = -A1 * m1 * c0s1st;
				strn(0, 1) =  s0 * c1 * st * (A0 * m1 - A1 * m0) / 2.0;
			}
			return false;
		}
		if (crdSpaceDim == 3)
		{
			f0 = phyConf->subConf[subConfigIndex]->descData.data(0);
			A0 = phyConf->subConf[subConfigIndex]->descData.data(3);
			m0 = phyConf->subConf[subConfigIndex]->descData.data(6);

			f1 = phyConf->subConf[subConfigIndex]->descData.data(1);
			A1 = phyConf->subConf[subConfigIndex]->descData.data(4);
			m1 = phyConf->subConf[subConfigIndex]->descData.data(7);

			f2 = phyConf->subConf[subConfigIndex]->descData.data(2);
			A2 = phyConf->subConf[subConfigIndex]->descData.data(5);
			m2 = phyConf->subConf[subConfigIndex]->descData.data(8);

			double x0 = X(0), mx0 = x0 * m0, c0 = cos(mx0), s0 = sin(mx0);
			double x1 = X(1), mx1 = x1 * m1, c1 = cos(mx1), s1 = sin(mx1);
			double x2 = X(2), mx2 = x2 * m2, c2 = cos(mx2), s2 = sin(mx2);
			double alphaT = alpha * T, ctAlpha = cos(alphaT) * alpha, st = sin(alphaT);
			double tmp0 = A0 * s0 * s1 * s2, tmp1 = A1 * c0 * c1 * s2, tmp2 = A2 * c0 * s1 * c2;
			if (computeU == true)
			{
				disp(0) = tmp0 * st;
				disp(1) = tmp1 * st;
				disp(2) = tmp2 * st;
			}
			if (computeV == true)
			{
				vel(0) = tmp0 * ctAlpha;
				vel(1) = tmp1 * ctAlpha;
				vel(2) = tmp2 * ctAlpha;
			}
			if (computeE == true)
			{
				double c0s1s2st = c0 * s1 * s2 * st;
				strn(0, 0) =  A0 * m0 * c0s1s2st;
				strn(1, 1) = -A1 * m1 * c0s1s2st;
				strn(2, 2) = -A2 * m2 * c0s1s2st;
				strn(0, 1) =  s0 * c1 * s2 * st * (A0 * m1 - A1 * m0) / 2.0;
				strn(0, 2) =  s0 * s1 * c2 * st * (A0 * m2 - A2 * m0) / 2.0;
				strn(1, 2) =  c0 * c1 * c2 * st * (A1 * m2 + A2 * m1) / 2.0;
			}
			return false;
		}
	}
	return false;
}

//	virtual fCrdSys getCrdSystemSpecificSourceTerm(const PhyDescriptor& desc) const;
//----------------------------------------------------------------------------------------------------------------------------------
//														MATERIAL PROPERTIES
//----------------------------------------------------------------------------------------------------------------------------------
bool getDescriptorSpecifiedMaterialPropertiesHomogeneous(SLSubConfig *slConfig, const PhyDescriptor& desc, map<int, vTensori>& materialFlags, int matLoadNumber, VECTOR& matData, IVECTOR& matFlags, map<int, vTensord>& materialProperties, int crdSpaceDim)
{
	if (matLoadNumber == 8) // exact solution smooth
	{
		E = 1.0;
		nu = 0.3;
		rho0 = 1.0;
		double cd = getCdIsotropic(rho0, nu, E, SL3DFull, 3);
		damping = 0.0; // 2 * cd * rho0;
	}
	return true;
}

//	virtual fCrdSys getCrdSystemSpecificMaterialProperties(const PhyDescriptor& desc) const;

//----------------------------------------------------------------------------------------------------------------------------------
//														MAX WAVE SPEED FOR A POINT
//----------------------------------------------------------------------------------------------------------------------------------
int SLSubConfig::getMaxExactSolution(int icbcNo, int& maxExactSourceTermOrder)
{
	if (icbcNo == 8)
		maxOrder = 3;
	return maxOrder;
}

void SLSubConfig::setMaxIC_BC_BF_orders(int& icMaxOrder, int & bcMaxOrder, int& bfMaxOrder)
{
	int icbcNo = descData.icbcNo;
	if (b_hasExactSolution == true)
	{
		int maxOrder, maxExactSourceTermOrder;
		maxOrder = getMaxExactSolution(icbcNo, maxExactSourceTermOrder);
		icMaxOrder = maxOrder;
		bcMaxOrder = maxOrder;
		bfMaxOrder = maxExactSourceTermOrder;
		return;
	}
	icMaxOrder = -1, bcMaxOrder = -1, bfMaxOrder = -1;
}

// exact solution
bool SLPhysics::computeExactSolutionIndividualPhysicsLevel(vector<PhyFldC>& exactFlds, int num, const PhyDescriptor& desc, ptCoords& crds, mapPfc2Td* dataPtr, int crdSpaceDim, VECTOR& X, double T, compT ct) const
{
	double rho = getRho0(desc, crds);
	double dampingF; 
	int icbc = phyConf->subConf[subConfigIndex]->descData.icbcNo;
	
	// other general loading:

	PhyFldC dispT(pfDisp, ct), plmT(pfplm, ct), velT(pfVel, ct), strnT(pfStrnL, ct), strsT(pfStrsL, ct), bfvT(pfVel, ctSource), bfpT(pfplm, ctSource);
	bool computeU = false, computeV = false, computePlm = false, computeS = false, computeE = false, compBv = false, compBp = false, compB = false, compDF = false;
//	bool computeU = (Find(exactFlds, dispT) >= 0), computeV = (Find(exactFlds, velT) >= 0);
//	bool computePlm = (Find(exactFlds, plmT)  >= 0), computeS = (Find(exactFlds, strsT)  >= 0), computeE = ((Find(exactFlds, strnT)  >= 0) || (computeS == true)), compBv = (Find(exactFlds, bfvT) >= 0), compBp = (Find(exactFlds, bfpT) >= 0), compB = (compBv || compBp);

	for (int i = 0; i < num; ++i)
	{
		phyFld fld = exactFlds[i].phyF;
		if (fld == pfDisp)
			computeU = true;
		else if (fld == pfplm)
		{
			computePlm = true;
			computeV = true;
		}
		else if (fld == pfStrsL)
		{
			computeS = true;
			computeE = true;
		}
		else if (fld == pfVel)
			computeV = true;
		else if (fld == pfStrnL)
			computeE = true;
		else if (fld == pfDampingF)
		{
			compDF = true;
			dampingF = getDampingCoefficient(desc, crds);
		}
		else if (exactFlds[i].cT = ctSource)
			compBv = true;
	}
/*	
	if (compBp == true)
	{
		cout << "exactFlds\n" << exactFlds << endl;
		cout << "computeU\t" << computeU << endl;
		cout << "computeV\t" << computeV << endl;
		cout << "computeE\t" << computeE << endl;
		cout << "compB\t" << compB << endl;
		THROW("should fix the function for when body force is needed\n");
	}
*/
	computeExactSolutionIndividualPhysicsLevelUVEB(icbc, desc, crds, (*dataPtr)[dispT], (*dataPtr)[velT], (*dataPtr)[strnT], (*dataPtr)[bfvT], crdSpaceDim, X, T, computeU, computeV, computeE, compBv);

//	bool computePS = true;
//	if (computePS == false)
//		return true;
	vTensord *vel = &(*dataPtr)[velT];
	if (computePlm == true)
	{
		vTensord *plm = &(*dataPtr)[plmT]; 
		plm->set_vTensor(1, crdSpaceDim, tenFULL);
		for (int i = 0; i < crdSpaceDim; ++i)
			(*plm)(i) = rho * (*vel)(i);
	}
//	if (compDF == true)
	{
		dampingF = getDampingCoefficient(desc, crds);
		PhyFldC dfT(pfDampingF, ct);
		vTensord *dmpingForce = &(*dataPtr)[dfT];
		dmpingForce->set_vTensor(1, crdSpaceDim, tenFULL);
		for (int i = 0; i < crdSpaceDim; ++i)
			(*dmpingForce)(i) = -dampingF * (*vel)(i);
	}
	if (computeS == true)
		TransferLinearStrn2StrsLinearSolidCartesianAxis(desc, crds, (*dataPtr)[strnT], (*dataPtr)[strsT]);
	return true;
}


//----------------------------------------------------------------------------------------------------------------------------------
//														INITIAL CONDITION	
//----------------------------------------------------------------------------------------------------------------------------------
void SLPhysics::getDescriptorPrescribedSpecificIC(const PhyDescriptor& desc, ptCoords& crds, const vector<PhyFldC>& pfcs, mapPfc2Td& data, int loadNumber, VECTOR& loadData, IVECTOR& loadFlags, intStage intS)
{
	if (phyConf->subConf[subConfigIndex]->b_hasExactSolution == true)
	{
		crds.setCartesianX();
		VECTOR X;
		double T = crds.XC.getTime();
		crds.XC.getSCrd(X);
		int crdSpaceDim = peParent->get_crdSpaceDim();
		bool computeU = true, computeV = true, computeE = true, compB = false;
		vTensord bodyForce;
		computeExactSolutionIndividualPhysicsLevelUVEB(loadNumber, desc, crds, data[disp], data[vel], data[strn], bodyForce, crdSpaceDim, X, T, computeU, computeV, computeE, compB);
# if PRINT_EXACT_DB
		db << "p \t" << this->peParent->patch->getID() << "\n";
		db << "ICX\n" << X << "\tT\t" << T << '\n';
		db << "disp\t" << data[disp] << "\tvel\t" << data[vel] << "\tstrn\t" << data[strn] << '\n';
#endif
		return;
	}
	else if (loadNumber >= 3) // Penny shape (4) Boeing (5), ..., (10) shock loading
	{
		data[disp].setValue(0);
		data[vel].setValue(0);
		data[strn].setValue(0);
		// set Displacement, Velocity, and Strain, ... here
	}
	else
		THROW("unknown loadNumber!\n");
	//// ....
	PhyFldC dispDT(pfDisp, ctDT);
	data[dispDT].reference(data[vel]);
}

//----------------------------------------------------------------------------------------------------------------------------------
//														BOUNDARY CONDITION	
//----------------------------------------------------------------------------------------------------------------------------------
void SLPhysics::getDescriptorPrescribedSpecificBC(const PhyDescriptor& desc, ptCoords& crds, const vector<PhyFldC>& pfcs, mapPfc2Td& data, mapPfc2Ti& flags, int loadNumber, VECTOR& loadData, IVECTOR& loadFlags, fCrdSys crdSys, BndryFT bndryT, intStage intS, int iVertical, const V2TENSOR& localCoordBasis) const
{
	if (phyConf->subConf[subConfigIndex]->b_hasExactSolution == true)
	{
		crds.setCartesianX();
		VECTOR X;
		double T = crds.XC.getTime();
		crds.XC.getSCrd(X);
		int crdSpaceDim = peParent->get_crdSpaceDim();
		vector<PhyFldC> exactFlds;
		if (computeU == true)
			exactFlds.push_back(disp);
		exactFlds.push_back(vel);
		compT ct = ctVal;
		int e_Index = 0;
		PhyDescriptor *descInteriorPtr = crds.getPhyDesc(e_Index);
		computeExactSolutionIndividualPhysicsLevel(exactFlds, 1, *descInteriorPtr, crds, &data, crdSpaceDim, X, T, ct);
		flags[vel].setValue(1);

#if PRINT_EXACT_DB
		db << "p \t" << this->peParent->patch->getID() << "\n";
		db << "BCX\n" << X << "\tT\t" << T << '\n';
		db << "flags[vel]\t" << flags[vel] << "\tvel\t" << data[vel];
		db << "flags[strs]\t" << flags[strs] << "\tstrs\t" << data[strs];
#endif
	}
}

fCrdSys SLPhysics::getCrdSystemSpecificBC(const PhyDescriptor& desc) const
{
	loadType genLoadT = phyConf->subConf[subConfigIndex]->descData.genLoadT;
	if (genLoadT == polyLoading)
	{
		map< PhyDescriptor, bool>::iterator iter = phyConf->subConf[subConfigIndex]->descData.polySolnFld->localCoordinate.find(desc);
		if (iter == phyConf->subConf[subConfigIndex]->descData.polySolnFld->localCoordinate.end())
			THROW("(desc == phyConf->subConf[subConfigIndex]->descData.polySolnFld->localCoordinate.end())!\n");
		if (iter->second == true)
			return rotSys;
		return cartSys; 
	}
	if (phyConf->subConf[subConfigIndex]->b_hasExactSolution == true)
	{
		if (genLoadT == regularLoading)
			return cartSys; 
		else if (genLoadT == sinosoidalLoading)
			return rotSys;
	}
	return rotSys;
}

//----------------------------------------------------------------------------------------------------------------------------------
//															SOURCE TERM	
//----------------------------------------------------------------------------------------------------------------------------------
// returns "true" if source term is nonzero
bool SLPhysics::getDescriptorPrescribedSpecificSourceTerm(const PhyDescriptor& desc, ptCoords& crds, int e_Index, rotT rT, const vector<PhyFldC>& pfcs, mapPfc2Td& data, int loadNumber, VECTOR& loadData, IVECTOR& loadFlags, fCrdSys crdSys, intStage intS, int iVertical) const
{
	if (phyConf->subConf[subConfigIndex]->b_hasExactSolution == true)
	{
		crds.setCartesianX();
		VECTOR X;
		double T = crds.XC.getTime();
		crds.XC.getSCrd(X);
		int crdSpaceDim = peParent->get_crdSpaceDim();
		bool computeU = false, computeV = false, computeE = false, compB = true;
		vTensord disp, vel, strn;
		bool ret = computeExactSolutionIndividualPhysicsLevelUVEB(loadNumber, desc, crds, disp, vel, strn, data[b], crdSpaceDim, X, T, computeU, computeV, computeE, compB);
#if PRINT_EXACT_DB
		db << "p \t" << this->peParent->patch->getID() << "\n";
		db << "SrcX\n" << X << "\tT\t" << T << "\tret\t" << ret << '\n';
		db << "data[b]\t" << data[b] << '\n';
#endif
		return ret;
	}
}
