#ifndef G_TIME__H
#define G_TIME__H


#define HAVE_BOOST_CHRONO_	1

#if HAVE_BOOST_CHRONO_
// #include </InstalledPackages/boost_1_52_0/boost/chrono/typeof/boost/chrono/chrono.hpp>
// #include </InstalledPackages/boost_1_52_0/boost/chrono/chrono_io.hpp>
// #include </InstalledPackages/boost_1_52_0/boost/chrono/duration.hpp>
// #include </InstalledPackages/boost_1_52_0/boost/chrono/time_point.hpp>
// #include </InstalledPackages/boost_1_52_0/boost/chrono/system_clocks.hpp>
// #include </InstalledPackages/boost_1_52_0/boost/chrono/process_cpu_clocks.hpp>

#include <boost/chrono.hpp>
// #include <boost/chrono/chrono_io.hpp>
// #include <boost/chrono/duration.hpp>
// #include <boost/chrono/time_point.hpp>
// #include <boost/chrono/system_clocks.hpp>
// #include <boost/chrono/process_cpu_clocks.hpp>
#if VCPP
#include </InstalledPackages/boost_1_52_0/boost/chrono/thread_clock.hpp> 
#endif
#endif

//InstalledPackages/boost_1_52_0/boost/chrono/typeof/boost/chrono

//#include "PhyTensor.h"

//#include </InstalledPackages/boost_1_52_0/boost/archive/text_oarchive.hpp>
//#include </InstalledPackages/boost_1_52_0/boost/archive/text_iarchive.hpp>
//#include </InstalledPackages/boost_1_52_0/boost/serialization/split_member.hpp>
//#include </InstalledPackages/boost_1_52_0/boost/serialization/base_object.hpp>
//#include </InstalledPackages/boost_1_52_0/boost/serialization/vector.hpp>
//#include </InstalledPackages/boost_1_52_0/boost/serialization/map.hpp>

/*
//TODO:  Make build system have this path!
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
*/

#include <iostream>
#include <vector>
#include <string>

typedef enum {gt_system, gt_steady, gt_highR, gt_thread, gt_realCPU, gt_userCPU, gt_systemCPU} gTimes_T;
#define SIZE_gTimes_T	7

class gTime
{
	friend std::ostream& operator << (std::ostream& output, gTime& dat);

public:
	gTime(bool setStart = false);
	void setZero();
	void setStartgTime();
	void setEndgTime();
	void print(std::ostream& output);
	void add(gTime& t2);

	// for incremental time measuring (on and off computation times are adeded between the following two commands:
	void setStartgTimeIncremental();
	void setEndgTimeIncremental();

//	vTensor<double> durations;	
	double durations[SIZE_gTimes_T];

	static int getClockNames(std::vector<std::string>& names);

	template <class Archive> 
	void serialize(Archive& ar, unsigned int version) 
	{
		for (int i = 0; i < SIZE_gTimes_T; ++i)
			ar & durations[i];
//		ar & durations;	
	}

private:
#if HAVE_BOOST_CHRONO_
	boost::chrono::system_clock::time_point stSystemC;
	boost::chrono::steady_clock::time_point stSteadyC;
	boost::chrono::high_resolution_clock::time_point stHighRC;
#if VCPP
	boost::chrono::thread_clock::time_point stThreadC; 
#endif
	boost::chrono::process_real_cpu_clock::time_point st_realCPUC;
	boost::chrono::process_user_cpu_clock::time_point st_userCPUC;
	boost::chrono::process_system_cpu_clock::time_point st_systemCPUC;
#endif
};

gTime::gTime(bool setStart)
{
    //	durations.set_vTensor(1, SIZE_gTimes_T, tenFULL);
	setZero();
	if (setStart == true)
		setStartgTime();
}

void gTime::setZero()
{
	for (int i = 0; i < SIZE_gTimes_T; ++i)
		durations[i] = 0.0;
    
    //	durations = 0.0;
}

void gTime::setStartgTime()
{
#if HAVE_BOOST_CHRONO_
	stSystemC = boost::chrono::system_clock::now();
	stSteadyC = boost::chrono::steady_clock::now();
	stHighRC = boost::chrono::high_resolution_clock::now();
#if VCPP
	stThreadC = boost::chrono::thread_clock::now();
#endif
	st_realCPUC = boost::chrono::process_real_cpu_clock::now();
	st_userCPUC = boost::chrono::process_user_cpu_clock::now();
	st_systemCPUC = boost::chrono::process_system_cpu_clock::now();
#endif
}

void gTime::setStartgTimeIncremental()
{
	setStartgTime();
}

void gTime::setEndgTimeIncremental()
{
#if HAVE_BOOST_CHRONO_
	boost::chrono::duration<double> sec;
	sec = boost::chrono::system_clock::now() - stSystemC;
	durations[gt_system] += sec.count();
    
	sec = boost::chrono::steady_clock::now() - stSteadyC;
	durations[gt_steady] += sec.count();
    
	sec = boost::chrono::high_resolution_clock::now() - stHighRC;
	durations[gt_highR] += sec.count();
    
#if VCPP
	sec = boost::chrono::thread_clock::now() - stThreadC;
	durations[gt_thread] += sec.count();
#endif
    
	sec = boost::chrono::process_real_cpu_clock::now() - st_realCPUC;
	durations[gt_realCPU] += sec.count();
    
	sec = boost::chrono::process_user_cpu_clock::now() - st_userCPUC;
	durations[gt_userCPU] += sec.count();
    
	sec = boost::chrono::process_system_cpu_clock::now() - st_systemCPUC;
	durations[gt_systemCPU] += sec.count();
#endif
}


void gTime::setEndgTime()
{
#if HAVE_BOOST_CHRONO_
	boost::chrono::duration<double> sec;
	sec = boost::chrono::system_clock::now() - stSystemC;
	durations[gt_system] = sec.count();
    
	sec = boost::chrono::steady_clock::now() - stSteadyC;
	durations[gt_steady] = sec.count();
    
	sec = boost::chrono::high_resolution_clock::now() - stHighRC;
	durations[gt_highR] = sec.count();
    
#if VCPP
	sec = boost::chrono::thread_clock::now() - stThreadC;
	durations[gt_thread] = sec.count();
#endif
    
	sec = boost::chrono::process_real_cpu_clock::now() - st_realCPUC;
	durations[gt_realCPU] = sec.count();
    
	sec = boost::chrono::process_user_cpu_clock::now() - st_userCPUC;
	durations[gt_userCPU] = sec.count();
    
	sec = boost::chrono::process_system_cpu_clock::now() - st_systemCPUC;
	durations[gt_systemCPU] = sec.count();
#endif
}

void gTime::print(std::ostream& output)
{
	output << "system\t" << durations[gt_system] << '\n';
	output << "steady\t" << durations[gt_steady] << '\n';
	output << "highR\t" << durations[gt_highR] << '\n';
	output << "thread\t" << durations[gt_thread] << '\n';
	output << "realCPU\t" << durations[gt_realCPU] << '\n';
	output << "userCPU\t" << durations[gt_userCPU] << '\n';
	output << "systemCPU\t" << durations[gt_systemCPU] << '\n';
}

int gTime::getClockNames(std::vector<std::string>& names)
{
	names.resize(7);
	names[0] = "system";
	names[1] = "steady";
	names[2] = "highR";
	names[3] = "thread";
	names[4] = "realCPU";
	names[5] = "userCPU";
	names[6] = "systemCPU";
	return 7;
}

void gTime::add(gTime& t2)
{
	for (int i = 0; i < SIZE_gTimes_T; ++i)
		durations[i] += t2.durations[i];
}

std::ostream& operator << (std::ostream& output, gTime& dat)
{
	dat.print(output);
	return output;
}

#endif