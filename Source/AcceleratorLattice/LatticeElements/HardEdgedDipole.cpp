/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HardEdgedDipole.H"
#include "Utils/WarpXUtil.H"
#include "Utils/TextMsg.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>

HardEdgedDipole::HardEdgedDipole ()
    : LatticeElementBase("dipole")
{
}

void
HardEdgedDipole::AddElement (amrex::ParmParse & pp_element, amrex::Real & z_location)
{
    using namespace amrex::literals;

    AddElementBase(pp_element, z_location);

    amrex::Real Ex = 0._rt;
    amrex::Real Ey = 0._rt;
    amrex::Real Bx = 0._rt;
    amrex::Real By = 0._rt;
    pp_element.query("Ex", Ex);
    pp_element.query("Ey", Ey);
    pp_element.query("Bx", Bx);
    pp_element.query("By", By);

    h_Ex.push_back(Ex);
    h_Ex.push_back(Ey);
    h_Bx.push_back(Bx);
    h_Ex.push_back(By);
}

void
HardEdgedDipole::WriteToDevice ()
{
    WriteToDeviceBase();

    d_Ex.resize(h_Ex.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Ex.begin(), h_Ex.end(), d_Ex.begin());
    d_Ey.resize(h_Ey.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Ey.begin(), h_Ey.end(), d_Ey.begin());
    d_Bx.resize(h_Bx.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_Bx.begin(), h_Bx.end(), d_Bx.begin());
    d_By.resize(h_By.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_By.begin(), h_By.end(), d_By.begin());
}

HardEdgedDipoleDevice
HardEdgedDipole::GetDeviceInstance () const
{
    HardEdgedDipoleDevice result;
    result.InitHardEdgedDipoleDevice(*this);
    return result;
}

void
HardEdgedDipoleDevice::InitHardEdgedDipoleDevice (HardEdgedDipole const& h_dipo)
{

    nelements = h_dipo.nelements;

    if (nelements == 0) return;

    d_zs_arr = h_dipo.d_zs.data();
    d_ze_arr = h_dipo.d_ze.data();

    d_Ex_arr = h_dipo.d_Ex.data();
    d_Ey_arr = h_dipo.d_Ey.data();
    d_Bx_arr = h_dipo.d_Bx.data();
    d_By_arr = h_dipo.d_By.data();

}
