/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy, Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FiniteDifferenceSolver.H"

#ifdef WARPX_DIM_RZ
#   include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#   include "FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "HybridPICModel/HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

using namespace amrex;

void FiniteDifferenceSolver::HybridPICEvolveEDisplacement (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Jfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jifield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jextfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::unique_ptr<amrex::MultiFab> const& rhofield,
    std::unique_ptr<amrex::MultiFab> const& Pefield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& edge_lengths,
    amrex::Real dt, int lev, HybridPICModel const* hybrid_model,
    const bool include_resistivity_term)
{
    // Select algorithm (The choice of algorithm is a runtime option,
    // but we compile code for each algorithm, using templates)
    if (m_fdtd_algo == ElectromagneticSolverAlgo::HybridPIC) {
#ifdef WARPX_DIM_RZ

        HybridPICEvolveEDisplacementCylindrical <CylindricalYeeAlgorithm> (
            Efield, Jfield, Jifield, Jextfield, Bfield, rhofield, Pefield,
            edge_lengths, lev, hybrid_model, include_resistivity_term
        );

#else

        HybridPICEvolveEDisplacementCartesian <CartesianYeeAlgorithm> (
            Efield, Jfield, Jifield, Jextfield, Bfield, rhofield, Pefield,
            edge_lengths, dt, lev, hybrid_model, include_resistivity_term
        );

#endif
    } else {
        amrex::Abort(Utils::TextMsg::Err(
            "HybridSolveE: The hybrid-PIC electromagnetic solver algorithm must be used"));
    }
}

#ifdef WARPX_DIM_RZ
template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICEvolveEDisplacementCylindrical (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jifield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jextfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::unique_ptr<amrex::MultiFab> const& rhofield,
    std::unique_ptr<amrex::MultiFab> const& Pefield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& edge_lengths,
    int lev, HybridPICModel const* hybrid_model,
    const bool include_resistivity_term )
{
#ifndef AMREX_USE_EB
    amrex::ignore_unused(edge_lengths);
#endif

    // Both steps below do not currently support m > 0 and should be
    // modified if such support wants to be added
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_nmodes == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;

    const bool include_hyper_resistivity_term = (eta_h > 0.0) && include_resistivity_term;

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Er_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Et_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jr_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jt_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Br_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& Bt_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;

    // Parameters for `interp` that maps from Yee to nodal mesh and back
    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // The E-field calculation is done in 2 steps:
    // 1) The J x B term is calculated on a nodal mesh in order to ensure
    //    energy conservation.
    // 2) The nodal E-field values are averaged onto the Yee grid and the
    //    electron pressure & resistivity terms are added (these terms are
    //    naturally located on the Yee grid).

    // Create a temporary multifab to hold the nodal E-field values
    // Note the multifab has 3 values for Ex, Ey and Ez which we can do here
    // since all three components will be calculated on the same grid.
    // Also note that enE_nodal_mf does not need to have any guard cells since
    // these values will be interpolated to the Yee mesh which is contained
    // by the nodal mesh.
    auto const& ba = convert(rhofield->boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield->DistributionMap(), 3, IntVect::TheZeroVector());

    // Loop through the grids, and over the tiles within each grid for the
    // initial, nodal calculation of E
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jt = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jir = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jit = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Jextr = Jextfield[0]->const_array(mfi);
        Array4<Real const> const& Jextt = Jextfield[1]->const_array(mfi);
        Array4<Real const> const& Jextz = Jextfield[2]->const_array(mfi);
        Array4<Real const> const& Br = Bfield[0]->const_array(mfi);
        Array4<Real const> const& Bt = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){

            // interpolate the total current to a nodal grid
            auto const jr_interp = Interp(Jr, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jt_interp = Interp(Jt, Jt_stag, nodal, coarsen, i, j, 0, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the ion current to a nodal grid
            auto const jir_interp = Interp(Jir, Jr_stag, nodal, coarsen, i, j, 0, 0);
            auto const jit_interp = Interp(Jit, Jt_stag, nodal, coarsen, i, j, 0, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, 0, 0);

            // interpolate the B field to a nodal grid
            auto const Br_interp = Interp(Br, Br_stag, nodal, coarsen, i, j, 0, 0);
            auto const Bt_interp = Interp(Bt, Bt_stag, nodal, coarsen, i, j, 0, 0);
            auto const Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, 0, 0);

            // calculate enE = (J - Ji) x B
            enE_nodal(i, j, 0, 0) = (
                (jt_interp - jit_interp - Jextt(i, j, 0)) * Bz_interp
                - (jz_interp - jiz_interp - Jextz(i, j, 0)) * Bt_interp
            );
            enE_nodal(i, j, 0, 1) = (
                (jz_interp - jiz_interp - Jextz(i, j, 0)) * Br_interp
                - (jr_interp - jir_interp - Jextr(i, j, 0)) * Bz_interp
            );
            enE_nodal(i, j, 0, 2) = (
                (jr_interp - jir_interp - Jextr(i, j, 0)) * Bt_interp
                - (jt_interp - jit_interp - Jextt(i, j, 0)) * Br_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    // Loop through the grids, and over the tiles within each grid again
    // for the Yee grid calculation of the E field
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = static_cast<Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Er = Efield[0]->array(mfi);
        Array4<Real> const& Et = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Jr = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jt = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield->const_array(mfi);
        Array4<Real> const& Pe = Pefield->array(mfi);

#ifdef AMREX_USE_EB
        amrex::Array4<amrex::Real> const& lr = edge_lengths[0]->array(mfi);
        amrex::Array4<amrex::Real> const& lt = edge_lengths[1]->array(mfi);
        amrex::Array4<amrex::Real> const& lz = edge_lengths[2]->array(mfi);
#endif

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_r = m_stencil_coefs_r.dataPtr();
        int const n_coefs_r = static_cast<int>(m_stencil_coefs_r.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        int const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        // Extract cylindrical specific parameters
        Real const dr = m_dr;
        Real const rmin = m_rmin;

        Box const& ter  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tet  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        amrex::ParallelFor(ter, tet, tez,

            // Er calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
#ifdef AMREX_USE_EB
                // Skip if this cell is fully covered by embedded boundaries
                if (lr(i, j, 0) <= 0) return;
#endif
                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (include_resistivity_term && resistivity_has_J_dependence) {
                    const Real jr_val = Interp(Jr, Jr_stag, Er_stag, coarsen, i, j, 0, 0);
                    const Real jt_val = Interp(Jt, Jt_stag, Er_stag, coarsen, i, j, 0, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Er_stag, coarsen, i, j, 0, 0);
                    jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure
                auto grad_Pe = T_Algo::UpwardDr(Pe, coefs_r, n_coefs_r, i, j, 0, 0);

                // interpolate the nodal neE values to the Yee grid
                auto enE_r = Interp(enE, nodal, Er_stag, coarsen, i, j, 0, 0);

                Er(i, j, 0) = (enE_r - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { Er(i, j, 0) += eta(rho_val, jtot_val) * Jr(i, j, 0); }

                if (include_hyper_resistivity_term) {
                    // r on cell-centered point (Jr is cell-centered in r)
                    Real const r = rmin + (i + 0.5_rt)*dr;

                    auto nabla2Jr = T_Algo::Dr_rDr_over_r(Jr, r, dr, coefs_r, n_coefs_r, i, j, 0, 0);
                    Er(i, j, 0) -= eta_h * nabla2Jr;
                }
            },

            // Et calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
#ifdef AMREX_USE_EB
                // In RZ Et is associated with a mesh node, so we need to check if the mesh node is covered
                amrex::ignore_unused(lt);
                if (lr(i, j, 0)<=0 || lr(i-1, j, 0)<=0 || lz(i, j-1, 0)<=0 || lz(i, j, 0)<=0) return;
#endif
                // r on a nodal grid (Et is nodal in r)
                Real const r = rmin + i*dr;
                // Mode m=0: // Ensure that Et remains 0 on axis
                if (r < 0.5_rt*dr) {
                    Et(i, j, 0, 0) = 0.;
                    return;
                }

                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Er_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (include_resistivity_term && resistivity_has_J_dependence) {
                    const Real jr_val = Interp(Jr, Jr_stag, Et_stag, coarsen, i, j, 0, 0);
                    const Real jt_val = Interp(Jt, Jt_stag, Et_stag, coarsen, i, j, 0, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Et_stag, coarsen, i, j, 0, 0);
                    jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure
                // -> d/dt = 0 for m = 0
                auto grad_Pe = 0.0_rt;

                // interpolate the nodal neE values to the Yee grid
                auto enE_t = Interp(enE, nodal, Et_stag, coarsen, i, j, 0, 1);

                Et(i, j, 0) = (enE_t - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { Et(i, j, 0) += eta(rho_val, jtot_val) * Jt(i, j, 0); }

                // Note: Hyper-resisitivity should be revisited here when modal decomposition is implemented
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
#ifdef AMREX_USE_EB
                // Skip field solve if this cell is fully covered by embedded boundaries
                if (lz(i,j,0) <= 0) { return; }
#endif
                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, 0, 0);

                // Interpolate current to appropriate staggering to match E field
                Real jtot_val = 0._rt;
                if (include_resistivity_term && resistivity_has_J_dependence) {
                    const Real jr_val = Interp(Jr, Jr_stag, Ez_stag, coarsen, i, j, 0, 0);
                    const Real jt_val = Interp(Jt, Jt_stag, Ez_stag, coarsen, i, j, 0, 0);
                    const Real jz_val = Interp(Jz, Jz_stag, Ez_stag, coarsen, i, j, 0, 0);
                    jtot_val = std::sqrt(jr_val*jr_val + jt_val*jt_val + jz_val*jz_val);
                }

                // safety condition since we divide by rho_val later
                if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Get the gradient of the electron pressure
                auto grad_Pe = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, 0, 0);

                // interpolate the nodal neE values to the Yee grid
                auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, 0, 2);

                Ez(i, j, 0) = (enE_z - grad_Pe) / rho_val;

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { Ez(i, j, 0) += eta(rho_val, jtot_val) * Jz(i, j, 0); }

                if (include_hyper_resistivity_term) {
                    auto nabla2Jz = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, 0, 0);
                    Ez(i, j, 0) -= eta_h * nabla2Jz;
                }
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#else

template<typename T_Algo>
void FiniteDifferenceSolver::HybridPICEvolveEDisplacementCartesian (
    std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jifield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jextfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Bfield,
    std::unique_ptr<amrex::MultiFab> const& rhofield,
    std::unique_ptr<amrex::MultiFab> const& Pefield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& edge_lengths,
    amrex::Real dt, int lev, HybridPICModel const* hybrid_model,
    const bool include_resistivity_term )
{
#ifndef AMREX_USE_EB
    amrex::ignore_unused(edge_lengths);
#endif

    // for the profiler
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    using namespace ablastr::coarsen::sample;

    auto& warpx = WarpX::GetInstance();

    // get hybrid model parameters
    const auto eta = hybrid_model->m_eta;
    const auto eta_h = hybrid_model->m_eta_h;
    const auto rho_floor = hybrid_model->m_n_floor * PhysConst::q_e;
    const auto resistivity_has_J_dependence = hybrid_model->m_resistivity_has_J_dependence;

    const bool include_hyper_resistivity_term = (eta_h > 0.) && include_resistivity_term;

    // Index type required for interpolating fields from their respective
    // staggering to the Ex, Ey, Ez locations
    amrex::GpuArray<int, 3> const& Ex_stag = hybrid_model->Ex_IndexType;
    amrex::GpuArray<int, 3> const& Ey_stag = hybrid_model->Ey_IndexType;
    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
    amrex::GpuArray<int, 3> const& Jx_stag = hybrid_model->Jx_IndexType;
    amrex::GpuArray<int, 3> const& Jy_stag = hybrid_model->Jy_IndexType;
    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
    amrex::GpuArray<int, 3> const& Bx_stag = hybrid_model->Bx_IndexType;
    amrex::GpuArray<int, 3> const& By_stag = hybrid_model->By_IndexType;
    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;

    // Parameters for `interp` that maps from Yee to nodal mesh and back
    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};

    // The E-field calculation is done in 2 steps:
    // 1) The J x B term is calculated on a nodal mesh in order to ensure
    //    energy conservation.
    // 2) The nodal E-field values are averaged onto the Yee grid and the
    //    electron pressure & resistivity terms are added (these terms are
    //    naturally located on the Yee grid).

    // Create a temporary multifab to hold the nodal E-field values
    // Note the multifab has 3 values for Ex, Ey and Ez which we can do here
    // since all three components will be calculated on the same grid.
    // Also note that enE_nodal_mf does not need to have any guard cells since
    // these values will be interpolated to the Yee mesh which is contained
    // by the nodal mesh.
    auto const& ba = convert(rhofield->boxArray(), IntVect::TheNodeVector());
    MultiFab enE_nodal_mf(ba, rhofield->DistributionMap(), 3, IntVect::TheZeroVector());

    MultiFab grad_Pe_x_mf(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, Efield[0]->nGrowVect());
    MultiFab grad_Pe_y_mf(Efield[1]->boxArray(), Efield[1]->DistributionMap(), 1, Efield[1]->nGrowVect());
    MultiFab grad_Pe_z_mf(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, Efield[2]->nGrowVect());

    MultiFab nabla2_J_x_mf(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, Efield[0]->nGrowVect());
    MultiFab nabla2_J_y_mf(Efield[1]->boxArray(), Efield[1]->DistributionMap(), 1, Efield[1]->nGrowVect());
    MultiFab nabla2_J_z_mf(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, Efield[2]->nGrowVect());

    // Loop through the grids, and over the tiles within each grid for the
    // initial, nodal calculation of E
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        Array4<Real> const& enE_nodal = enE_nodal_mf.array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& Jix = Jifield[0]->const_array(mfi);
        Array4<Real const> const& Jiy = Jifield[1]->const_array(mfi);
        Array4<Real const> const& Jiz = Jifield[2]->const_array(mfi);
        Array4<Real const> const& Jextx = Jextfield[0]->const_array(mfi);
        Array4<Real const> const& Jexty = Jextfield[1]->const_array(mfi);
        Array4<Real const> const& Jextz = Jextfield[2]->const_array(mfi);
        Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const& By = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // interpolate the total current to a nodal grid
            auto const jx_interp = Interp(Jx, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jy_interp = Interp(Jy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jz_interp = Interp(Jz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the ion current to a nodal grid
            auto const jix_interp = Interp(Jix, Jx_stag, nodal, coarsen, i, j, k, 0);
            auto const jiy_interp = Interp(Jiy, Jy_stag, nodal, coarsen, i, j, k, 0);
            auto const jiz_interp = Interp(Jiz, Jz_stag, nodal, coarsen, i, j, k, 0);

            // interpolate the B field to a nodal grid
            auto const Bx_interp = Interp(Bx, Bx_stag, nodal, coarsen, i, j, k, 0);
            auto const By_interp = Interp(By, By_stag, nodal, coarsen, i, j, k, 0);
            auto const Bz_interp = Interp(Bz, Bz_stag, nodal, coarsen, i, j, k, 0);

            // calculate enE = (J - Ji) x B
            enE_nodal(i, j, k, 0) = (
                (jy_interp - jiy_interp - Jexty(i, j, k)) * Bz_interp
                - (jz_interp - jiz_interp - Jextz(i, j, k)) * By_interp
            );
            enE_nodal(i, j, k, 1) = (
                (jz_interp - jiz_interp - Jextz(i, j, k)) * Bx_interp
                - (jx_interp - jix_interp - Jextx(i, j, k)) * Bz_interp
            );
            enE_nodal(i, j, k, 2) = (
                (jx_interp - jix_interp - Jextx(i, j, k)) * By_interp
                - (jy_interp - jiy_interp - Jexty(i, j, k)) * Bx_interp
            );
        });

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    #ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(grad_Pe_x_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        Array4<Real> const& Pe = Pefield->array(mfi);
        Array4<Real> const& grad_Pe_x = grad_Pe_x_mf.array(mfi);
        Array4<Real> const& grad_Pe_y = grad_Pe_y_mf.array(mfi);
        Array4<Real> const& grad_Pe_z = grad_Pe_z_mf.array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real> const& nabla2_J_x = nabla2_J_x_mf.array(mfi);
        Array4<Real> const& nabla2_J_y = nabla2_J_y_mf.array(mfi);
        Array4<Real> const& nabla2_J_z = nabla2_J_z_mf.array(mfi);

        // Extract stencil coefficients
        Real const * const AMREX_RESTRICT coefs_x = m_stencil_coefs_x.dataPtr();
        auto const n_coefs_x = static_cast<int>(m_stencil_coefs_x.size());
        Real const * const AMREX_RESTRICT coefs_y = m_stencil_coefs_y.dataPtr();
        auto const n_coefs_y = static_cast<int>(m_stencil_coefs_y.size());
        Real const * const AMREX_RESTRICT coefs_z = m_stencil_coefs_z.dataPtr();
        auto const n_coefs_z = static_cast<int>(m_stencil_coefs_z.size());

        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the nodal E field
        amrex::ParallelFor(tex, tey, tez, 
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                grad_Pe_x(i,j,k) = T_Algo::UpwardDx(Pe, coefs_x, n_coefs_x, i, j, k);
                nabla2_J_x(i,j,k) = T_Algo::Dxx(Jx, coefs_x, n_coefs_x, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                grad_Pe_y(i,j,k) = T_Algo::UpwardDy(Pe, coefs_y, n_coefs_y, i, j, k);
                nabla2_J_y(i,j,k) = T_Algo::Dyy(Jy, coefs_y, n_coefs_y, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                grad_Pe_z(i,j,k) = T_Algo::UpwardDz(Pe, coefs_z, n_coefs_z, i, j, k);
                nabla2_J_z(i,j,k) = T_Algo::Dzz(Jz, coefs_z, n_coefs_z, i, j, k);
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

    grad_Pe_x_mf.FillBoundary(warpx.Geom(lev).periodicity());
    grad_Pe_y_mf.FillBoundary(warpx.Geom(lev).periodicity());
    grad_Pe_z_mf.FillBoundary(warpx.Geom(lev).periodicity());
    nabla2_J_x_mf.FillBoundary(warpx.Geom(lev).periodicity());
    nabla2_J_y_mf.FillBoundary(warpx.Geom(lev).periodicity());
    nabla2_J_z_mf.FillBoundary(warpx.Geom(lev).periodicity());

    // Loop through the grids, and over the tiles within each grid again
    // for the Yee grid calculation of the E field
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Extract field data for this grid/tile
        Array4<Real> const& Ex = Efield[0]->array(mfi);
        Array4<Real> const& Ey = Efield[1]->array(mfi);
        Array4<Real> const& Ez = Efield[2]->array(mfi);
        Array4<Real const> const& Bx = Bfield[0]->const_array(mfi);
        Array4<Real const> const& By = Bfield[1]->const_array(mfi);
        Array4<Real const> const& Bz = Bfield[2]->const_array(mfi);
        Array4<Real const> const& Jx = Jfield[0]->const_array(mfi);
        Array4<Real const> const& Jy = Jfield[1]->const_array(mfi);
        Array4<Real const> const& Jz = Jfield[2]->const_array(mfi);
        Array4<Real const> const& enE = enE_nodal_mf.const_array(mfi);
        Array4<Real const> const& rho = rhofield->const_array(mfi);
        Array4<Real const> const& grad_Pe_x = grad_Pe_x_mf.const_array(mfi);
        Array4<Real const> const& grad_Pe_y = grad_Pe_y_mf.const_array(mfi);
        Array4<Real const> const& grad_Pe_z = grad_Pe_z_mf.const_array(mfi);
        Array4<Real const> const& nabla2_J_x = nabla2_J_x_mf.const_array(mfi);
        Array4<Real const> const& nabla2_J_y = nabla2_J_y_mf.const_array(mfi);
        Array4<Real const> const& nabla2_J_z = nabla2_J_z_mf.const_array(mfi);

#ifdef AMREX_USE_EB
        amrex::Array4<amrex::Real> const& lx = edge_lengths[0]->array(mfi);
        amrex::Array4<amrex::Real> const& ly = edge_lengths[1]->array(mfi);
        amrex::Array4<amrex::Real> const& lz = edge_lengths[2]->array(mfi);
#endif

        Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
        Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
        Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

        // Loop over the cells and update the E field
        amrex::ParallelFor(tex, tey, tez,

            // Ex calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef AMREX_USE_EB
                // Skip if this cell is fully covered by embedded boundaries
                if (lx(i, j, k) <= 0) return;
#endif
                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ex_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jx_val = Interp(Jx, Jx_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real jy_val = Interp(Jy, Jy_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);

                // safety condition since we divide by rho_val later
                // This is now ignored since displacement current introduced
                //if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Ex staggering 
                auto grad_Pe_x_val = grad_Pe_x(i, j, k);
                auto grad_Pe_y_val = Interp(grad_Pe_y, Ey_stag, Ex_stag, coarsen, i, j, k, 0);
                auto grad_Pe_z_val = Interp(grad_Pe_z, Ez_stag, Ex_stag, coarsen, i, j, k, 0);

                // interpolate the nodal enE values to the Yee grid for Ex staggering
                auto enE_x = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 0);
                auto enE_y = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 1);
                auto enE_z = Interp(enE, nodal, Ex_stag, coarsen, i, j, k, 2);

                // Interpolate B values to Ex staggering
                auto Bx_val = Interp(Bx, Bx_stag, Ex_stag, coarsen, i, j, k, 0);
                auto By_val = Interp(By, By_stag, Ex_stag, coarsen, i, j, k, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Ex_stag, coarsen, i, j, k, 0);

                // Store old E values
                const Real Exo = Ex(i,j,k);
                const Real Eyo = Interp(Ey, Ey_stag, Ex_stag, coarsen, i, j, k, 0);
                const Real Ezo = Interp(Ez, Ez_stag, Ex_stag, coarsen, i, j, k, 0);

                Real Jtilde_x = enE_x - grad_Pe_x_val;
                Real Jtilde_y = enE_y - grad_Pe_y_val;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_x += rho_val * eta_val * jx_val; 
                    Jtilde_y += rho_val * eta_val * jy_val;
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_x_val = nabla2_J_x(i, j, k);
                    auto nabla2_J_y_val = Interp(nabla2_J_y, Ey_stag, Ex_stag, coarsen, i, j, k, 0);
                    auto nabla2_J_z_val = Interp(nabla2_J_z, Ez_stag, Ex_stag, coarsen, i, j, k, 0);

                    Jtilde_x -= rho_val * eta_h * nabla2_J_x_val;
                    Jtilde_y -= rho_val * eta_h * nabla2_J_y_val;
                    Jtilde_z -= rho_val * eta_h * nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                Jtilde_x -= 0.5_rt * rho_val + PhysConst::ep0 / dt * (eta_val*Exo + Bz_val*Eyo - By_val*Ezo); 
                Jtilde_y -= PhysConst::ep0 / dt * (eta_val*Eyo - Bz_val*Exo + Bx_val*Ezo);
                Jtilde_z -= PhysConst::ep0 / dt * (eta_val*Ezo + By_val*Exo - Bx_val*Eyo);

                // Calculate inverse for semi-implicit advance
                Real d = 0.5_rt * rho_val - PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + Bx_val*Bx_val + By_val*By_val + Bz_val*Bz_val));

                Ex(i,j,k) = coeff * (
                    (d*d + Bx_val*Bx_val) * Jtilde_x
                    + (Bx_val*By_val + d*Bz_val) * Jtilde_y
                    + (Bx_val*Bz_val - d*By_val) * Jtilde_z
                    );
            },

            // Ey calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef AMREX_USE_EB
                // Skip field solve if this cell is fully covered by embedded boundaries
#ifdef WARPX_DIM_3D
                if (ly(i,j,k) <= 0) { return; }
#elif defined(WARPX_DIM_XZ)
                //In XZ Ey is associated with a mesh node, so we need to check if the mesh node is covered
                amrex::ignore_unused(ly);
                if (lx(i, j, k)<=0 || lx(i-1, j, k)<=0 || lz(i, j-1, k)<=0 || lz(i, j, k)<=0) { return; }
#endif
#endif
                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ey_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jx_val = Interp(Jx, Jx_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real jy_val = Interp(Jy, Jy_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);
                
                // safety condition since we divide by rho_val later
                // This is now ignored since displacement current introduced
                //if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Ex staggering 
                auto grad_Pe_x_val = Interp(grad_Pe_x, Ex_stag, Ey_stag, coarsen, i, j, k, 0);
                auto grad_Pe_y_val = grad_Pe_y(i, j, k);
                auto grad_Pe_z_val = Interp(grad_Pe_z, Ez_stag, Ey_stag, coarsen, i, j, k, 0);

                // interpolate the nodal enE values to the Yee grid for Ex staggering
                auto enE_x = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 0);
                auto enE_y = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 1);
                auto enE_z = Interp(enE, nodal, Ey_stag, coarsen, i, j, k, 2);

                // Interpolate B values to Ex staggering
                auto Bx_val = Interp(Bx, Bx_stag, Ey_stag, coarsen, i, j, k, 0);
                auto By_val = Interp(By, By_stag, Ey_stag, coarsen, i, j, k, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Ey_stag, coarsen, i, j, k, 0);

                // Store old E values
                const Real Exo = Interp(Ex, Ex_stag, Ey_stag, coarsen, i, j, k, 0);
                const Real Eyo = Ey(i,j,k);
                const Real Ezo = Interp(Ez, Ez_stag, Ey_stag, coarsen, i, j, k, 0);

                Real Jtilde_x = enE_x - grad_Pe_x_val;
                Real Jtilde_y = enE_y - grad_Pe_y_val;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_x += rho_val * eta_val * jx_val; 
                    Jtilde_y += rho_val * eta_val * jy_val;
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_x_val = Interp(nabla2_J_x, Ex_stag, Ey_stag, coarsen, i, j, k, 0);
                    auto nabla2_J_y_val = nabla2_J_y(i, j, k);
                    auto nabla2_J_z_val = Interp(nabla2_J_z, Ez_stag, Ey_stag, coarsen, i, j, k, 0);

                    Jtilde_x -= rho_val * eta_h * nabla2_J_x_val;
                    Jtilde_y -= rho_val * eta_h * nabla2_J_y_val;
                    Jtilde_z -= rho_val * eta_h * nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                Jtilde_x -= PhysConst::ep0 / dt * (eta_val*Exo + Bz_val*Eyo - By_val*Ezo); 
                Jtilde_y -= 0.5_rt * rho_val + PhysConst::ep0 / dt * (eta_val*Eyo - Bz_val*Exo + Bx_val*Ezo);
                Jtilde_z -= PhysConst::ep0 / dt * (eta_val*Ezo + By_val*Exo - Bx_val*Eyo);

                // Calculate inverse for semi-implicit advance
                Real d = 0.5_rt * rho_val - PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + Bx_val*Bx_val + By_val*By_val + Bz_val*Bz_val));

                Ey(i,j,k) = coeff * (
                    (Bx_val*By_val - d*Bz_val) * Jtilde_x
                    + (d*d + By_val*By_val) * Jtilde_y
                    + (By_val*Bz_val + d*Bx_val) * Jtilde_z
                    );
            },

            // Ez calculation
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
#ifdef AMREX_USE_EB
                // Skip field solve if this cell is fully covered by embedded boundaries
                if (lz(i,j,k) <= 0) { return; }
#endif
                
                // Interpolate to get the appropriate charge density in space
                Real rho_val = Interp(rho, nodal, Ez_stag, coarsen, i, j, k, 0);

                // Interpolate current to appropriate staggering to match E field
                const Real jx_val = Interp(Jx, Jx_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real jy_val = Interp(Jy, Jy_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real jz_val = Interp(Jz, Jz_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real jtot_val = std::sqrt(jx_val*jx_val + jy_val*jy_val + jz_val*jz_val);

                // safety condition since we divide by rho_val later
                // This is now ignored since displacement current introduced
                //if (rho_val < rho_floor) { rho_val = rho_floor; }

                // Interpolate the electron pressure gradient to Ex staggering 
                auto grad_Pe_x_val = Interp(grad_Pe_x, Ex_stag, Ez_stag, coarsen, i, j, k, 0);
                auto grad_Pe_y_val = Interp(grad_Pe_y, Ey_stag, Ez_stag, coarsen, i, j, k, 0);
                auto grad_Pe_z_val = grad_Pe_z(i, j, k);

                // interpolate the nodal enE values to the Yee grid for Ex staggering
                auto enE_x = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 0);
                auto enE_y = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 1);
                auto enE_z = Interp(enE, nodal, Ez_stag, coarsen, i, j, k, 2);

                // Interpolate B values to Ex staggering
                auto Bx_val = Interp(Bx, Bx_stag, Ez_stag, coarsen, i, j, k, 0);
                auto By_val = Interp(By, By_stag, Ez_stag, coarsen, i, j, k, 0);
                auto Bz_val = Interp(Bz, Bz_stag, Ez_stag, coarsen, i, j, k, 0);

                // Store old E values
                const Real Exo = Interp(Ex, Ex_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real Eyo = Interp(Ey, Ey_stag, Ez_stag, coarsen, i, j, k, 0);
                const Real Ezo = Ez(i,j,k);

                Real Jtilde_x = enE_x - grad_Pe_x_val;
                Real Jtilde_y = enE_y - grad_Pe_y_val;
                Real Jtilde_z = enE_z - grad_Pe_z_val;

                const auto eta_val = eta(rho_val, jtot_val);

                // Add resistivity only if E field value is used to update B
                if (include_resistivity_term) { 
                    Jtilde_x += rho_val * eta_val * jx_val; 
                    Jtilde_y += rho_val * eta_val * jy_val;
                    Jtilde_z += rho_val * eta_val * jz_val;
                }

                if (include_hyper_resistivity_term) {
                    auto nabla2_J_x_val = Interp(nabla2_J_x, Ex_stag, Ez_stag, coarsen, i, j, k, 0);
                    auto nabla2_J_y_val = Interp(nabla2_J_y, Ey_stag, Ez_stag, coarsen, i, j, k, 0);
                    auto nabla2_J_z_val = nabla2_J_z(i, j, k);

                    Jtilde_x -= rho_val * eta_h * nabla2_J_x_val;
                    Jtilde_y -= rho_val * eta_h * nabla2_J_y_val;
                    Jtilde_z -= rho_val * eta_h * nabla2_J_z_val;
                }

                // Include explicit terms for displacement current
                Jtilde_x -= PhysConst::ep0 / dt * (eta_val*Exo + Bz_val*Eyo - By_val*Ezo); 
                Jtilde_y -= PhysConst::ep0 / dt * (eta_val*Eyo - Bz_val*Exo + Bx_val*Ezo);
                Jtilde_z -= 0.5_rt * rho_val + PhysConst::ep0 / dt * (eta_val*Ezo + By_val*Exo - Bx_val*Eyo);

                // Calculate inverse for semi-implicit advance
                Real d = 0.5_rt * rho_val - PhysConst::ep0 / dt * eta_val;
                Real coeff = 1.0_rt / (d * (d*d + Bx_val*Bx_val + By_val*By_val + Bz_val*Bz_val));

                Ez(i,j,k) = coeff * (
                    (Bx_val*Bz_val + d*By_val) * Jtilde_x
                    + (By_val*Bz_val - d*Bx_val) * Jtilde_y
                    + (d*d + Bz_val*Bz_val) * Jtilde_z
                    );
            }
        );

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}
#endif
