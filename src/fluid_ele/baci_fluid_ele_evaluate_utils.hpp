/*----------------------------------------------------------------------*/
/*! \file

\brief Utilities for the fluid element evaluation

\level 1


*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_FLUID_ELE_EVALUATE_UTILS_HPP
#define FOUR_C_FLUID_ELE_EVALUATE_UTILS_HPP

#include "baci_config.hpp"

#include "baci_discretization_fem_general_extract_values.hpp"
#include "baci_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "baci_discretization_fem_general_utils_nurbs_shapefunctions.hpp"
#include "baci_discretization_geometry_position_array.hpp"
#include "baci_fluid_ele.hpp"
#include "baci_fluid_ele_action.hpp"
#include "baci_fluid_ele_parameter_std.hpp"
#include "baci_inpar_turbulence.hpp"
#include "baci_lib_element_integration_select.hpp"
#include "baci_mat_newtonianfluid.hpp"
#include "baci_mat_sutherland.hpp"
#include "baci_nurbs_discret.hpp"

BACI_NAMESPACE_OPEN

namespace FLD
{
  //-----------------------------------------------------------------------
  // turbulence-related methods
  //-----------------------------------------------------------------------



  /*!
    \brief same as above for low-Mach-number flow
   */
  // this routine is supposed to move to fluid_ele_calc_general_service.cpp and use the methods
  // provided there move it if you are using it in a similar way as CalcChannelStatistics
  template <int iel>
  void f3_calc_loma_means(DRT::ELEMENTS::Fluid* ele, DRT::Discretization& discretization,
      std::vector<double>& velocitypressure, std::vector<double>& temperature,
      Teuchos::ParameterList& params, const double eosfac)
  {
    // get view of solution vector
    CORE::LINALG::Matrix<4 * iel, 1> velpre(velocitypressure.data(), true);
    CORE::LINALG::Matrix<4 * iel, 1> temp(temperature.data(), true);

    // set element data
    const CORE::FE::CellType distype = ele->Shape();

    // the plane normal tells you in which plane the integration takes place
    const int normdirect = params.get<int>("normal direction to homogeneous plane");

    // the vector planes contains the coordinates of the homogeneous planes (in
    // wall normal direction)
    Teuchos::RCP<std::vector<double>> planes =
        params.get<Teuchos::RCP<std::vector<double>>>("coordinate vector for hom. planes");

    // get the pointers to the solution vectors
    Teuchos::RCP<std::vector<double>> sumarea =
        params.get<Teuchos::RCP<std::vector<double>>>("element layer area");

    Teuchos::RCP<std::vector<double>> sumu =
        params.get<Teuchos::RCP<std::vector<double>>>("mean velocity u");
    Teuchos::RCP<std::vector<double>> sumv =
        params.get<Teuchos::RCP<std::vector<double>>>("mean velocity v");
    Teuchos::RCP<std::vector<double>> sumw =
        params.get<Teuchos::RCP<std::vector<double>>>("mean velocity w");
    Teuchos::RCP<std::vector<double>> sump =
        params.get<Teuchos::RCP<std::vector<double>>>("mean pressure p");
    Teuchos::RCP<std::vector<double>> sumrho =
        params.get<Teuchos::RCP<std::vector<double>>>("mean density rho");
    Teuchos::RCP<std::vector<double>> sumT =
        params.get<Teuchos::RCP<std::vector<double>>>("mean temperature T");
    Teuchos::RCP<std::vector<double>> sumrhou =
        params.get<Teuchos::RCP<std::vector<double>>>("mean momentum rho*u");
    Teuchos::RCP<std::vector<double>> sumrhouT =
        params.get<Teuchos::RCP<std::vector<double>>>("mean rho*u*T");

    Teuchos::RCP<std::vector<double>> sumsqu =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value u^2");
    Teuchos::RCP<std::vector<double>> sumsqv =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value v^2");
    Teuchos::RCP<std::vector<double>> sumsqw =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value w^2");
    Teuchos::RCP<std::vector<double>> sumsqp =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value p^2");
    Teuchos::RCP<std::vector<double>> sumsqrho =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value rho^2");
    Teuchos::RCP<std::vector<double>> sumsqT =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value T^2");

    Teuchos::RCP<std::vector<double>> sumuv =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value uv");
    Teuchos::RCP<std::vector<double>> sumuw =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value uw");
    Teuchos::RCP<std::vector<double>> sumvw =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value vw");
    Teuchos::RCP<std::vector<double>> sumuT =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value uT");
    Teuchos::RCP<std::vector<double>> sumvT =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value vT");
    Teuchos::RCP<std::vector<double>> sumwT =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value wT");

    // get node coordinates of element
    CORE::LINALG::Matrix<3, iel> xyze;
    DRT::Node** nodes = ele->Nodes();
    for (int inode = 0; inode < iel; inode++)
    {
      const auto& x = nodes[inode]->X();
      xyze(0, inode) = x[0];
      xyze(1, inode) = x[1];
      xyze(2, inode) = x[2];
    }

    if (distype == CORE::FE::CellType::hex8 || distype == CORE::FE::CellType::hex27 ||
        distype == CORE::FE::CellType::hex20)
    {
      double min = xyze(normdirect, 0);
      double max = xyze(normdirect, 0);

      // set maximum and minimum value in wall normal direction
      for (int inode = 0; inode < iel; inode++)
      {
        if (min > xyze(normdirect, inode)) min = xyze(normdirect, inode);
        if (max < xyze(normdirect, inode)) max = xyze(normdirect, inode);
      }

      // determine the ids of the homogeneous planes intersecting this element
      std::set<int> planesinele;
      for (unsigned nplane = 0; nplane < planes->size(); ++nplane)
      {
        // get all available wall normal coordinates
        for (int nn = 0; nn < iel; ++nn)
        {
          if (min - 2e-9 < (*planes)[nplane] && max + 2e-9 > (*planes)[nplane])
            planesinele.insert(nplane);
        }
      }

      // remove lowest layer from planesinele to avoid double calculations. This is not done
      // for the first level (index 0) --- if deleted, shift the first integration point in
      // wall normal direction
      // the shift depends on the number of sampling planes in the element
      double shift = 0;

      // set the number of planes which cut the element
      const int numplanesinele = planesinele.size();

      if (*planesinele.begin() != 0)
      {
        // this is not an element of the lowest element layer
        planesinele.erase(planesinele.begin());

        shift = 2.0 / (static_cast<double>(numplanesinele - 1));
      }
      else
      {
        // this is an element of the lowest element layer. Increase the counter
        // in order to compute the total number of elements in one layer
        int* count = params.get<int*>("count processed elements");

        (*count)++;
      }

      // determine the orientation of the rst system compared to the xyz system
      int elenormdirect = -1;
      bool upsidedown = false;
      // the only thing of interest is how normdirect is oriented in the
      // element coordinate system
      if (xyze(normdirect, 4) - xyze(normdirect, 0) > 2e-9)
      {
        // t aligned
        elenormdirect = 2;
      }
      else if (xyze(normdirect, 3) - xyze(normdirect, 0) > 2e-9)
      {
        // s aligned
        elenormdirect = 1;
      }
      else if (xyze(normdirect, 1) - xyze(normdirect, 0) > 2e-9)
      {
        // r aligned
        elenormdirect = 0;
      }
      else if (xyze(normdirect, 4) - xyze(normdirect, 0) < -2e-9)
      {
        // -t aligned
        elenormdirect = 2;
        upsidedown = true;
      }
      else if (xyze(normdirect, 3) - xyze(normdirect, 0) < -2e-9)
      {
        // -s aligned
        elenormdirect = 1;
        upsidedown = true;
      }
      else if (xyze(normdirect, 1) - xyze(normdirect, 0) < -2e-9)
      {
        // -r aligned
        elenormdirect = 0;
        upsidedown = true;
      }
      else
      {
        dserror(
            "cannot determine orientation of plane normal in local coordinate system of element");
      }
      std::vector<int> inplanedirect;
      {
        std::set<int> inplanedirectset;
        for (int i = 0; i < 3; ++i)
        {
          inplanedirectset.insert(i);
        }
        inplanedirectset.erase(elenormdirect);

        for (std::set<int>::iterator id = inplanedirectset.begin(); id != inplanedirectset.end();
             ++id)
        {
          inplanedirect.push_back(*id);
        }
      }

      // allocate vector for shapefunctions
      CORE::LINALG::Matrix<iel, 1> funct;
      // allocate vector for shapederivatives
      CORE::LINALG::Matrix<3, iel> deriv;
      // space for the jacobian
      CORE::LINALG::Matrix<3, 3> xjm;

      // get the quad9 gaussrule for the in-plane integration
      const CORE::FE::IntegrationPoints2D intpoints(CORE::FE::GaussRule2D::quad_9point);

      // a hex8 element has two levels, the hex20 and hex27 element have three layers to sample
      // (now we allow even more)
      double layershift = 0;

      // loop all levels in element
      for (std::set<int>::const_iterator id = planesinele.begin(); id != planesinele.end(); ++id)
      {
        // reset temporary values
        double area = 0;

        double ubar = 0;
        double vbar = 0;
        double wbar = 0;
        double pbar = 0;
        double rhobar = 0;
        double Tbar = 0;
        double rhoubar = 0;
        double rhouTbar = 0;

        double usqbar = 0;
        double vsqbar = 0;
        double wsqbar = 0;
        double psqbar = 0;
        double rhosqbar = 0;
        double Tsqbar = 0;

        double uvbar = 0;
        double uwbar = 0;
        double vwbar = 0;
        double uTbar = 0;
        double vTbar = 0;
        double wTbar = 0;

        // get the integration point in wall normal direction
        double e[3];

        e[elenormdirect] = -1.0 + shift + layershift;
        if (upsidedown) e[elenormdirect] *= -1;

        // start loop over integration points in layer
        for (int iquad = 0; iquad < intpoints.nquad; iquad++)
        {
          // get the other gauss point coordinates
          for (int i = 0; i < 2; ++i)
          {
            e[inplanedirect[i]] = intpoints.qxg[iquad][i];
          }

          // compute the shape function values
          CORE::FE::shape_function_3D(funct, e[0], e[1], e[2], distype);
          CORE::FE::shape_function_3D_deriv1(deriv, e[0], e[1], e[2], distype);

          // get transposed Jacobian matrix and determinant
          //
          //        +-            -+ T      +-            -+
          //        | dx   dx   dx |        | dx   dy   dz |
          //        | --   --   -- |        | --   --   -- |
          //        | dr   ds   dt |        | dr   dr   dr |
          //        |              |        |              |
          //        | dy   dy   dy |        | dx   dy   dz |
          //        | --   --   -- |   =    | --   --   -- |
          //        | dr   ds   dt |        | ds   ds   ds |
          //        |              |        |              |
          //        | dz   dz   dz |        | dx   dy   dz |
          //        | --   --   -- |        | --   --   -- |
          //        | dr   ds   dt |        | dt   dt   dt |
          //        +-            -+        +-            -+
          //
          // The Jacobian is computed using the formula
          //
          //            +-----
          //   dx_j(r)   \      dN_k(r)
          //   -------  = +     ------- * (x_j)_k
          //    dr_i     /       dr_i       |
          //            +-----    |         |
          //            node k    |         |
          //                  derivative    |
          //                   of shape     |
          //                   function     |
          //                           component of
          //                          node coordinate
          //
          xjm.MultiplyNT(deriv, xyze);

          // we assume that every plane parallel to the wall is preserved
          // hence we can compute the jacobian determinant of the 2d cutting
          // element by replacing max-min by one on the diagonal of the
          // jacobi matrix (the two non-diagonal elements are zero)
          if (xjm(elenormdirect, normdirect) < 0)
            xjm(elenormdirect, normdirect) = -1.0;
          else
            xjm(elenormdirect, normdirect) = 1.0;

          const double det = xjm(0, 0) * xjm(1, 1) * xjm(2, 2) + xjm(0, 1) * xjm(1, 2) * xjm(2, 0) +
                             xjm(0, 2) * xjm(1, 0) * xjm(2, 1) - xjm(0, 2) * xjm(1, 1) * xjm(2, 0) -
                             xjm(0, 0) * xjm(1, 2) * xjm(2, 1) - xjm(0, 1) * xjm(1, 0) * xjm(2, 2);

          // check for degenerated elements
          if (det <= 0.0)
            dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", ele->Id(), det);

          // interpolated values at gausspoints
          double ugp = 0;
          double vgp = 0;
          double wgp = 0;
          double pgp = 0;
          double rhogp = 0;
          double Tgp = 0;
          double rhougp = 0;
          double rhouTgp = 0;

          // the computation of this jacobian determinant from the 3d
          // mapping is based on the assumption that we do not deform
          // our elements in wall normal direction!
          const double fac = det * intpoints.qwgt[iquad];

          // increase area of cutting plane in element
          area += fac;

          for (int inode = 0; inode < iel; inode++)
          {
            int finode = inode * 4;

            ugp += funct(inode) * velpre(finode++);
            vgp += funct(inode) * velpre(finode++);
            wgp += funct(inode) * velpre(finode++);
            pgp += funct(inode) * velpre(finode);
            Tgp += funct(inode) * temp(finode);
          }
          rhogp = eosfac / Tgp;
          rhouTgp = eosfac * ugp;
          rhougp = rhouTgp / Tgp;

          // add contribution to integral
          double dubar = ugp * fac;
          double dvbar = vgp * fac;
          double dwbar = wgp * fac;
          double dpbar = pgp * fac;
          double drhobar = rhogp * fac;
          double dTbar = Tgp * fac;
          double drhoubar = rhougp * fac;
          double drhouTbar = rhouTgp * fac;

          ubar += dubar;
          vbar += dvbar;
          wbar += dwbar;
          pbar += dpbar;
          rhobar += drhobar;
          Tbar += dTbar;
          rhoubar += drhoubar;
          rhouTbar += drhouTbar;

          usqbar += ugp * dubar;
          vsqbar += vgp * dvbar;
          wsqbar += wgp * dwbar;
          psqbar += pgp * dpbar;
          rhosqbar += rhogp * drhobar;
          Tsqbar += Tgp * dTbar;

          uvbar += ugp * dvbar;
          uwbar += ugp * dwbar;
          vwbar += vgp * dwbar;
          uTbar += ugp * dTbar;
          vTbar += vgp * dTbar;
          wTbar += wgp * dTbar;
        }  // end loop integration points

        // add increments from this layer to processor local vectors
        (*sumarea)[*id] += area;

        (*sumu)[*id] += ubar;
        (*sumv)[*id] += vbar;
        (*sumw)[*id] += wbar;
        (*sump)[*id] += pbar;
        (*sumrho)[*id] += rhobar;
        (*sumT)[*id] += Tbar;
        (*sumrhou)[*id] += rhoubar;
        (*sumrhouT)[*id] += rhouTbar;

        (*sumsqu)[*id] += usqbar;
        (*sumsqv)[*id] += vsqbar;
        (*sumsqw)[*id] += wsqbar;
        (*sumsqp)[*id] += psqbar;
        (*sumsqrho)[*id] += rhosqbar;
        (*sumsqT)[*id] += Tsqbar;

        (*sumuv)[*id] += uvbar;
        (*sumuw)[*id] += uwbar;
        (*sumvw)[*id] += vwbar;
        (*sumuT)[*id] += uTbar;
        (*sumvT)[*id] += vTbar;
        (*sumwT)[*id] += wTbar;

        // jump to the next layer in the element.
        // in case of an hex8 element, the two coordinates are -1 and 1(+2)
        // for quadratic elements with three sample planes, we have -1,0(+1),1(+2)

        layershift += 2.0 / (static_cast<double>(numplanesinele - 1));
      }
    }
    else
      dserror("Unknown element type for low-Mach-number mean value evaluation\n");

    return;
  }  // DRT::ELEMENTS::Fluid::f3_calc_loma_means


  /*!
    \brief same as above for turbulent scalar transport
   */
  // this routine is supposed to move to fluid_ele_calc_general_service.cpp and use the methods
  // provided there move it if you are using it in a similar way as CalcChannelStatistics
  template <int iel>
  void f3_calc_scatra_means(DRT::ELEMENTS::Fluid* ele, DRT::Discretization& discretization,
      std::vector<double>& velocitypressure, std::vector<double>& scalar,
      Teuchos::ParameterList& params)
  {
    // get view of solution vector
    CORE::LINALG::Matrix<4 * iel, 1> velpre(velocitypressure.data(), true);
    CORE::LINALG::Matrix<4 * iel, 1> phi(scalar.data(), true);

    // set element data
    const CORE::FE::CellType distype = ele->Shape();

    // the plane normal tells you in which plane the integration takes place
    const int normdirect = params.get<int>("normal direction to homogeneous plane");

    // the vector planes contains the coordinates of the homogeneous planes (in
    // wall normal direction)
    Teuchos::RCP<std::vector<double>> planes =
        params.get<Teuchos::RCP<std::vector<double>>>("coordinate vector for hom. planes");

    // get the pointers to the solution vectors
    Teuchos::RCP<std::vector<double>> sumarea =
        params.get<Teuchos::RCP<std::vector<double>>>("element layer area");

    Teuchos::RCP<std::vector<double>> sumu =
        params.get<Teuchos::RCP<std::vector<double>>>("mean velocity u");
    Teuchos::RCP<std::vector<double>> sumv =
        params.get<Teuchos::RCP<std::vector<double>>>("mean velocity v");
    Teuchos::RCP<std::vector<double>> sumw =
        params.get<Teuchos::RCP<std::vector<double>>>("mean velocity w");
    Teuchos::RCP<std::vector<double>> sump =
        params.get<Teuchos::RCP<std::vector<double>>>("mean pressure p");
    Teuchos::RCP<std::vector<double>> sumphi =
        params.get<Teuchos::RCP<std::vector<double>>>("mean scalar phi");

    Teuchos::RCP<std::vector<double>> sumsqu =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value u^2");
    Teuchos::RCP<std::vector<double>> sumsqv =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value v^2");
    Teuchos::RCP<std::vector<double>> sumsqw =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value w^2");
    Teuchos::RCP<std::vector<double>> sumsqp =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value p^2");
    Teuchos::RCP<std::vector<double>> sumsqphi =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value phi^2");

    Teuchos::RCP<std::vector<double>> sumuv =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value uv");
    Teuchos::RCP<std::vector<double>> sumuw =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value uw");
    Teuchos::RCP<std::vector<double>> sumvw =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value vw");
    Teuchos::RCP<std::vector<double>> sumuphi =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value uphi");
    Teuchos::RCP<std::vector<double>> sumvphi =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value vphi");
    Teuchos::RCP<std::vector<double>> sumwphi =
        params.get<Teuchos::RCP<std::vector<double>>>("mean value wphi");

    // get node coordinates of element
    CORE::LINALG::Matrix<3, iel> xyze;
    DRT::Node** nodes = ele->Nodes();
    for (int inode = 0; inode < iel; inode++)
    {
      const auto& x = nodes[inode]->X();
      xyze(0, inode) = x[0];
      xyze(1, inode) = x[1];
      xyze(2, inode) = x[2];
    }

    if (distype == CORE::FE::CellType::hex8 || distype == CORE::FE::CellType::hex27 ||
        distype == CORE::FE::CellType::hex20)
    {
      double min = xyze(normdirect, 0);
      double max = xyze(normdirect, 0);

      // set maximum and minimum value in wall normal direction
      for (int inode = 0; inode < iel; inode++)
      {
        if (min > xyze(normdirect, inode)) min = xyze(normdirect, inode);
        if (max < xyze(normdirect, inode)) max = xyze(normdirect, inode);
      }

      // determine the ids of the homogeneous planes intersecting this element
      std::set<int> planesinele;
      for (unsigned nplane = 0; nplane < planes->size(); ++nplane)
      {
        // get all available wall normal coordinates
        for (int nn = 0; nn < iel; ++nn)
        {
          if (min - 2e-9 < (*planes)[nplane] && max + 2e-9 > (*planes)[nplane])
            planesinele.insert(nplane);
        }
      }

      // remove lowest layer from planesinele to avoid double calculations. This is not done
      // for the first level (index 0) --- if deleted, shift the first integration point in
      // wall normal direction
      // the shift depends on the number of sampling planes in the element
      double shift = 0;

      // set the number of planes which cut the element
      const int numplanesinele = planesinele.size();

      if (*planesinele.begin() != 0)
      {
        // this is not an element of the lowest element layer
        planesinele.erase(planesinele.begin());

        shift = 2.0 / (static_cast<double>(numplanesinele - 1));
      }
      else
      {
        // this is an element of the lowest element layer. Increase the counter
        // in order to compute the total number of elements in one layer
        int* count = params.get<int*>("count processed elements");

        (*count)++;
      }

      // determine the orientation of the rst system compared to the xyz system
      int elenormdirect = -1;
      bool upsidedown = false;
      // the only thing of interest is how normdirect is oriented in the
      // element coordinate system
      if (xyze(normdirect, 4) - xyze(normdirect, 0) > 2e-9)
      {
        // t aligned
        elenormdirect = 2;
        std::cout << "upsidedown false" << &std::endl;
      }
      else if (xyze(normdirect, 3) - xyze(normdirect, 0) > 2e-9)
      {
        // s aligned
        elenormdirect = 1;
      }
      else if (xyze(normdirect, 1) - xyze(normdirect, 0) > 2e-9)
      {
        // r aligned
        elenormdirect = 0;
      }
      else if (xyze(normdirect, 4) - xyze(normdirect, 0) < -2e-9)
      {
        std::cout << xyze(normdirect, 4) - xyze(normdirect, 0) << &std::endl;
        // -t aligned
        elenormdirect = 2;
        upsidedown = true;
        std::cout << "upsidedown true" << &std::endl;
      }
      else if (xyze(normdirect, 3) - xyze(normdirect, 0) < -2e-9)
      {
        // -s aligned
        elenormdirect = 1;
        upsidedown = true;
      }
      else if (xyze(normdirect, 1) - xyze(normdirect, 0) < -2e-9)
      {
        // -r aligned
        elenormdirect = 0;
        upsidedown = true;
      }
      else
      {
        dserror(
            "cannot determine orientation of plane normal in local coordinate system of element");
      }
      std::vector<int> inplanedirect;
      {
        std::set<int> inplanedirectset;
        for (int i = 0; i < 3; ++i)
        {
          inplanedirectset.insert(i);
        }
        inplanedirectset.erase(elenormdirect);

        for (std::set<int>::iterator id = inplanedirectset.begin(); id != inplanedirectset.end();
             ++id)
        {
          inplanedirect.push_back(*id);
        }
      }

      // allocate vector for shapefunctions
      CORE::LINALG::Matrix<iel, 1> funct;
      // allocate vector for shapederivatives
      CORE::LINALG::Matrix<3, iel> deriv;
      // space for the jacobian
      CORE::LINALG::Matrix<3, 3> xjm;

      // get the quad9 gaussrule for the in-plane integration
      const CORE::FE::IntegrationPoints2D intpoints(CORE::FE::GaussRule2D::quad_9point);

      // a hex8 element has two levels, the hex20 and hex27 element have three layers to sample
      // (now we allow even more)
      double layershift = 0;

      // loop all levels in element
      for (std::set<int>::const_iterator id = planesinele.begin(); id != planesinele.end(); ++id)
      {
        // reset temporary values
        double area = 0;

        double ubar = 0;
        double vbar = 0;
        double wbar = 0;
        double pbar = 0;
        double phibar = 0;

        double usqbar = 0;
        double vsqbar = 0;
        double wsqbar = 0;
        double psqbar = 0;
        double phisqbar = 0;

        double uvbar = 0;
        double uwbar = 0;
        double vwbar = 0;
        double uphibar = 0;
        double vphibar = 0;
        double wphibar = 0;

        // get the integration point in wall normal direction
        double e[3];

        e[elenormdirect] = -1.0 + shift + layershift;
        if (upsidedown) e[elenormdirect] *= -1;

        // start loop over integration points in layer
        for (int iquad = 0; iquad < intpoints.nquad; iquad++)
        {
          // get the other gauss point coordinates
          for (int i = 0; i < 2; ++i)
          {
            e[inplanedirect[i]] = intpoints.qxg[iquad][i];
          }

          // compute the shape function values
          CORE::FE::shape_function_3D(funct, e[0], e[1], e[2], distype);
          CORE::FE::shape_function_3D_deriv1(deriv, e[0], e[1], e[2], distype);

          // get transposed Jacobian matrix and determinant
          //
          //        +-            -+ T      +-            -+
          //        | dx   dx   dx |        | dx   dy   dz |
          //        | --   --   -- |        | --   --   -- |
          //        | dr   ds   dt |        | dr   dr   dr |
          //        |              |        |              |
          //        | dy   dy   dy |        | dx   dy   dz |
          //        | --   --   -- |   =    | --   --   -- |
          //        | dr   ds   dt |        | ds   ds   ds |
          //        |              |        |              |
          //        | dz   dz   dz |        | dx   dy   dz |
          //        | --   --   -- |        | --   --   -- |
          //        | dr   ds   dt |        | dt   dt   dt |
          //        +-            -+        +-            -+
          //
          // The Jacobian is computed using the formula
          //
          //            +-----
          //   dx_j(r)   \      dN_k(r)
          //   -------  = +     ------- * (x_j)_k
          //    dr_i     /       dr_i       |
          //            +-----    |         |
          //            node k    |         |
          //                  derivative    |
          //                   of shape     |
          //                   function     |
          //                           component of
          //                          node coordinate
          //
          xjm.MultiplyNT(deriv, xyze);

          // we assume that every plane parallel to the wall is preserved
          // hence we can compute the jacobian determinant of the 2d cutting
          // element by replacing max-min by one on the diagonal of the
          // jacobi matrix (the two non-diagonal elements are zero)
          if (xjm(elenormdirect, normdirect) < 0)
            xjm(elenormdirect, normdirect) = -1.0;
          else
            xjm(elenormdirect, normdirect) = 1.0;

          const double det = xjm(0, 0) * xjm(1, 1) * xjm(2, 2) + xjm(0, 1) * xjm(1, 2) * xjm(2, 0) +
                             xjm(0, 2) * xjm(1, 0) * xjm(2, 1) - xjm(0, 2) * xjm(1, 1) * xjm(2, 0) -
                             xjm(0, 0) * xjm(1, 2) * xjm(2, 1) - xjm(0, 1) * xjm(1, 0) * xjm(2, 2);

          // check for degenerated elements
          if (det <= 0.0)
            dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", ele->Id(), det);

          // interpolated values at gausspoints
          double ugp = 0;
          double vgp = 0;
          double wgp = 0;
          double pgp = 0;
          double phigp = 0;

          // the computation of this jacobian determinant from the 3d
          // mapping is based on the assumption that we do not deform
          // our elements in wall normal direction!
          const double fac = det * intpoints.qwgt[iquad];

          // increase area of cutting plane in element
          area += fac;

          for (int inode = 0; inode < iel; inode++)
          {
            int finode = inode * 4;

            ugp += funct(inode) * velpre(finode++);
            vgp += funct(inode) * velpre(finode++);
            wgp += funct(inode) * velpre(finode++);
            pgp += funct(inode) * velpre(finode);
            phigp += funct(inode) * phi(finode);
          }

          // add contribution to integral
          double dubar = ugp * fac;
          double dvbar = vgp * fac;
          double dwbar = wgp * fac;
          double dpbar = pgp * fac;
          double dphibar = phigp * fac;

          ubar += dubar;
          vbar += dvbar;
          wbar += dwbar;
          pbar += dpbar;
          phibar += dphibar;

          usqbar += ugp * dubar;
          vsqbar += vgp * dvbar;
          wsqbar += wgp * dwbar;
          psqbar += pgp * dpbar;
          phisqbar += phigp * dphibar;

          uvbar += ugp * dvbar;
          uwbar += ugp * dwbar;
          vwbar += vgp * dwbar;
          uphibar += ugp * dphibar;
          vphibar += vgp * dphibar;
          wphibar += wgp * dphibar;
        }  // end loop integration points

        // add increments from this layer to processor local vectors
        (*sumarea)[*id] += area;

        (*sumu)[*id] += ubar;
        (*sumv)[*id] += vbar;
        (*sumw)[*id] += wbar;
        (*sump)[*id] += pbar;
        (*sumphi)[*id] += phibar;

        (*sumsqu)[*id] += usqbar;
        (*sumsqv)[*id] += vsqbar;
        (*sumsqw)[*id] += wsqbar;
        (*sumsqp)[*id] += psqbar;
        (*sumsqphi)[*id] += phisqbar;

        (*sumuv)[*id] += uvbar;
        (*sumuw)[*id] += uwbar;
        (*sumvw)[*id] += vwbar;
        (*sumuphi)[*id] += uphibar;
        (*sumvphi)[*id] += vphibar;
        (*sumwphi)[*id] += wphibar;

        // jump to the next layer in the element.
        // in case of an hex8 element, the two coordinates are -1 and 1(+2)
        // for quadratic elements with three sample planes, we have -1,0(+1),1(+2)

        layershift += 2.0 / (static_cast<double>(numplanesinele - 1));
      }
    }
    else
      dserror("Unknown element type for turbulent passive scalar mean value evaluation\n");

    return;
  }  // DRT::ELEMENTS::Fluid::f3_calc_scatra_means


  /*!
    \brief one point integration of convolutions with box filter
    function. The filter function is not normalized yet, this has
    to be done as a 'postprocessing' step after the patchvolume
    is calculated, i.e. after the volume contribution of all
    elements adjacent to a node has been added to the
    normalisation constant.

    See reference

    A.E. Tejada-Martinez, K.E. Jansen
    Spatial test filters for dynamic model large-eddy simulation with
    finite elements.
    Communications in numerical methods in engineering, 2000

    for details

    \param evelaf                        (in )  : nodal velocities
    \param vel_hat                       (out)  : integrated convolution
                                                  with velocity
    \param reystr_hat                    (out)  : integrated convolution
                                                  with reynolds stresses
    \param modeled_stress_grid_scale_hat (out)  : integrated convolution
                                                  with subgrid scale
                                                  stress tensor model
    \param volume                        (out)  : element volume
    \return void

   */
  template <int iel>
  void f3_apply_box_filter(DRT::ELEMENTS::Fluid* ele, DRT::ELEMENTS::FluidEleParameterStd* fldpara,
      std::vector<double>& myvel, std::vector<double>& mytemp, const double thermpress,
      Teuchos::RCP<std::vector<double>> vel_hat, Teuchos::RCP<std::vector<double>> densvel_hat,
      Teuchos::RCP<std::vector<std::vector<double>>> reynoldsstress_hat,
      Teuchos::RCP<std::vector<std::vector<double>>> modeled_subgrid_stress, double& volume,
      double& dens_hat, double& dens_strainrate_hat, double& expression_hat, double& alpha2_hat,
      Teuchos::RCP<std::vector<std::vector<double>>> strainrate_hat,
      Teuchos::RCP<std::vector<std::vector<double>>> alphaij_hat)
  {
    // number of spatial dimensions is always 3
    const int NSD = 3;

    // alloc a fixed size array for nodal velocities and temperature
    CORE::LINALG::Matrix<NSD, iel> evel;
    CORE::LINALG::Matrix<1, iel> etemp;

    // wrap matrix objects in fixed-size arrays
    CORE::LINALG::Matrix<(NSD + 1) * iel, 1> myvelvec(myvel.data(), true);
    CORE::LINALG::Matrix<iel, 1> mytempvec(mytemp.data(), true);

    // split velocity and throw away  pressure, insert into element array
    for (int i = 0; i < iel; ++i)
    {
      int fi = 4 * i;

      evel(0, i) = myvelvec(fi++);
      evel(1, i) = myvelvec(fi++);
      evel(2, i) = myvelvec(fi);

      etemp(0, i) = mytempvec(i);
    }

    // set element data
    const CORE::FE::CellType distype = ele->Shape();

    // allocate arrays for shape functions, derivatives and the transposed jacobian
    CORE::LINALG::Matrix<iel, 1> funct;
    CORE::LINALG::Matrix<NSD, NSD> xjm;
    CORE::LINALG::Matrix<NSD, NSD> xji;
    CORE::LINALG::Matrix<NSD, iel> deriv;
    CORE::LINALG::Matrix<NSD, iel> derxy;

    // useful variables
    CORE::LINALG::Matrix<NSD, 1> velint;
    double vdiv = 0.0;
    CORE::LINALG::Matrix<NSD, NSD> vderxy;
    CORE::LINALG::Matrix<NSD, NSD> epsilon;
    double rateofstrain = 0.0;

    // get node coordinates of element
    CORE::LINALG::Matrix<NSD, iel> xyze;
    for (int inode = 0; inode < iel; inode++)
    {
      xyze(0, inode) = ele->Nodes()[inode]->X()[0];
      xyze(1, inode) = ele->Nodes()[inode]->X()[1];
      xyze(2, inode) = ele->Nodes()[inode]->X()[2];
    }

    // get gauss rule: we use a one-point rule here
    CORE::FE::GaussRule3D integrationrule_filter = CORE::FE::GaussRule3D::hex_1point;
    switch (distype)
    {
      case CORE::FE::CellType::hex8:
        integrationrule_filter = CORE::FE::GaussRule3D::hex_1point;
        break;
      case CORE::FE::CellType::tet4:
        integrationrule_filter = CORE::FE::GaussRule3D::tet_1point;
        break;
      case CORE::FE::CellType::tet10:
      case CORE::FE::CellType::hex20:
      case CORE::FE::CellType::hex27:
        dserror("the box filtering operation is only permitted for linear elements\n");
        break;
      default:
        dserror("invalid discretization type for fluid3");
    }

    // gaussian points
    const CORE::FE::IntegrationPoints3D intpoints_onepoint(integrationrule_filter);

    // shape functions and derivs at element center
    const double e1 = intpoints_onepoint.qxg[0][0];
    const double e2 = intpoints_onepoint.qxg[0][1];
    const double e3 = intpoints_onepoint.qxg[0][2];
    const double wquad = intpoints_onepoint.qwgt[0];

    CORE::FE::shape_function_3D(funct, e1, e2, e3, distype);
    CORE::FE::shape_function_3D_deriv1(deriv, e1, e2, e3, distype);

    // get Jacobian matrix and determinant

    for (int nn = 0; nn < 3; ++nn)
    {
      for (int rr = 0; rr < 3; ++rr)
      {
        xjm(nn, rr) = deriv(nn, 0) * xyze(rr, 0);
        for (int mm = 1; mm < iel; ++mm)
        {
          xjm(nn, rr) += deriv(nn, mm) * xyze(rr, mm);
        }
      }
    }
    const double det = xjm(0, 0) * xjm(1, 1) * xjm(2, 2) + xjm(0, 1) * xjm(1, 2) * xjm(2, 0) +
                       xjm(0, 2) * xjm(1, 0) * xjm(2, 1) - xjm(0, 2) * xjm(1, 1) * xjm(2, 0) -
                       xjm(0, 0) * xjm(1, 2) * xjm(2, 1) - xjm(0, 1) * xjm(1, 0) * xjm(2, 2);

    /*
      Use the Jacobian and the known derivatives in element coordinate
      directions on the right hand side to compute the derivatives in
      global coordinate directions

            +-                 -+     +-    -+      +-    -+
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |     | ---- |      | ---- |
            |  dr    dr    dr   |     |  dx  |      |  dr  |
            |                   |     |      |      |      |
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |  *  | ---- |   =  | ---- | for all k
            |  ds    ds    ds   |     |  dy  |      |  ds  |
            |                   |     |      |      |      |
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |     | ---- |      | ---- |
            |  dt    dt    dt   |     |  dz  |      |  dt  |
            +-                 -+     +-    -+      +-    -+

    */
    xji(0, 0) = (xjm(1, 1) * xjm(2, 2) - xjm(2, 1) * xjm(1, 2)) / det;
    xji(1, 0) = (-xjm(1, 0) * xjm(2, 2) + xjm(2, 0) * xjm(1, 2)) / det;
    xji(2, 0) = (xjm(1, 0) * xjm(2, 1) - xjm(2, 0) * xjm(1, 1)) / det;
    xji(0, 1) = (-xjm(0, 1) * xjm(2, 2) + xjm(2, 1) * xjm(0, 2)) / det;
    xji(1, 1) = (xjm(0, 0) * xjm(2, 2) - xjm(2, 0) * xjm(0, 2)) / det;
    xji(2, 1) = (-xjm(0, 0) * xjm(2, 1) + xjm(2, 0) * xjm(0, 1)) / det;
    xji(0, 2) = (xjm(0, 1) * xjm(1, 2) - xjm(1, 1) * xjm(0, 2)) / det;
    xji(1, 2) = (-xjm(0, 0) * xjm(1, 2) + xjm(1, 0) * xjm(0, 2)) / det;
    xji(2, 2) = (xjm(0, 0) * xjm(1, 1) - xjm(1, 0) * xjm(0, 1)) / det;

    // compute global derivates
    for (int nn = 0; nn < NSD; ++nn)
    {
      for (int rr = 0; rr < iel; ++rr)
      {
        derxy(nn, rr) = deriv(0, rr) * xji(nn, 0);
        for (int mm = 1; mm < NSD; ++mm)
        {
          derxy(nn, rr) += deriv(mm, rr) * xji(nn, mm);
        }
      }
    }


    // get material at gauss point
    double dens = 0.0;
    Teuchos::RCP<MAT::Material> material = ele->Material();
    if (material->MaterialType() == INPAR::MAT::m_fluid)
    {
      const MAT::NewtonianFluid* actmat = static_cast<const MAT::NewtonianFluid*>(material.get());
      dens = actmat->Density();
    }
    else if (material->MaterialType() == INPAR::MAT::m_sutherland)
    {
      const MAT::Sutherland* actmat = static_cast<const MAT::Sutherland*>(material.get());

      // compute temperature at gauss point
      double temp = 0.0;
      for (int rr = 0; rr < iel; rr++) temp += funct(rr, 0) * etemp(0, rr);

      // compute density based on temperature
      dens = actmat->ComputeDensity(temp, thermpress);
    }

    // get velocities (n+alpha_F/1,i) at integration point
    //
    //                   +-----
    //       n+af/1       \                  n+af/1
    //    vel      (x) =   +      N (x) * vel
    //                    /        j         j
    //                   +-----
    //                   node j
    //
    // CORE::LINALG::Matrix<NSD,1> velint;
    for (int rr = 0; rr < NSD; ++rr)
    {
      velint(rr) = funct(0) * evel(rr, 0);
      for (int mm = 1; mm < iel; ++mm)
      {
        velint(rr) += funct(mm) * evel(rr, mm);
      }
    }

    if (fldpara->TurbModAction() == INPAR::FLUID::dynamic_smagorinsky)
    {
      // get velocity (n+alpha_F/1,i) derivatives at integration point
      //
      //       n+af/1      +-----  dN (x)
      //   dvel      (x)    \        k         n+af/1
      //   ------------- =   +     ------ * vel
      //        dx          /        dx        k
      //          j        +-----      j
      //                   node k
      //
      // j : direction of derivative x/y/z
      //
      // CORE::LINALG::Matrix<NSD,NSD> vderxy;
      for (int nn = 0; nn < NSD; ++nn)
      {
        for (int rr = 0; rr < NSD; ++rr)
        {
          vderxy(nn, rr) = derxy(rr, 0) * evel(nn, 0);
          for (int mm = 1; mm < iel; ++mm)
          {
            vderxy(nn, rr) += derxy(rr, mm) * evel(nn, mm);
          }
        }
      }

      // divergence of velocity
      vdiv = vderxy(0, 0) + vderxy(1, 1) + vderxy(2, 2);

      // get rate-of deformation tensor
      /*
                                +-     n+af/1          n+af/1    -+
              / h \       1.0   |  dvel_i    (x)   dvel_j    (x)  |
         eps | u   |    = --- * |  ------------- + -------------  |
              \   / ij    2.0   |       dx              dx        |
                                +-        j               i      -+
      */
      // CORE::LINALG::Matrix<NSD,NSD> epsilon;
      for (int nn = 0; nn < NSD; ++nn)
      {
        for (int rr = 0; rr < NSD; ++rr)
        {
          epsilon(nn, rr) = 0.5 * (vderxy(nn, rr) + vderxy(rr, nn));
        }
      }

      //
      // modeled part of subgrid scale stresses
      //
      /*    +-                                 -+ 1
            |          / h \           / h \    | -         / h \
            | 2 * eps | u   |   * eps | u   |   | 2  * eps | u   |
            |          \   / kl        \   / kl |           \   / ij
            +-                                 -+

            |                                   |
            +-----------------------------------+
                 'resolved' rate of strain
      */
      // double rateofstrain
      for (int rr = 0; rr < NSD; rr++)
      {
        for (int mm = 0; mm < NSD; mm++)
        {
          rateofstrain += epsilon(rr, mm) * epsilon(rr, mm);
        }
      }
      rateofstrain *= 2.0;
      rateofstrain = sqrt(rateofstrain);
    }


    //--------------------------------------------------
    // one point integrations

    // determine contribution to patch volume
    volume = wquad * det;

    if (not(fldpara->TurbModAction() == INPAR::FLUID::dynamic_vreman))
    {
      for (int rr = 0; rr < NSD; ++rr)
      {
        double tmp = velint(rr) * volume;

        // add contribution to integral over velocities
        (*vel_hat)[rr] += tmp;
        // add contribution to integral over dens times velocity
        if (fldpara->TurbModAction() == INPAR::FLUID::dynamic_smagorinsky and
            fldpara->PhysicalType() == INPAR::FLUID::loma)
          (*densvel_hat)[rr] += dens * tmp;

        // add contribution to integral over reynolds stresses
        for (int nn = 0; nn < NSD; ++nn)
        {
          (*reynoldsstress_hat)[rr][nn] += dens * tmp * velint(nn);
        }
      }
    }


    if (fldpara->TurbModAction() == INPAR::FLUID::dynamic_smagorinsky)
    {
      // add contribution to integral over the modeled part of subgrid
      // scale stresses
      double rateofstrain_volume = dens * rateofstrain * volume;
      for (int rr = 0; rr < NSD; ++rr)
      {
        for (int nn = 0; nn < NSD; ++nn)
        {
          (*modeled_subgrid_stress)[rr][nn] += rateofstrain_volume * epsilon(rr, nn);
          if (fldpara->PhysicalType() == INPAR::FLUID::loma and nn == rr)
            (*modeled_subgrid_stress)[rr][nn] -= 1.0 / 3.0 * rateofstrain_volume * vdiv;
        }
      }

      // add additional scalar quantities for loma
      // i.e., filtered density and filtered density times strainrate^2
      if (fldpara->PhysicalType() == INPAR::FLUID::loma)
      {
        dens_hat = dens * volume;
        dens_strainrate_hat = dens * volume * rateofstrain * rateofstrain;
      }
    }


    if (fldpara->TurbModAction() == INPAR::FLUID::dynamic_vreman)
    {
      // In the literature about the Vreman model, the indices i and j are swapped compared to the
      // standard definition in baci. All variables used for the Vreman model are used as in
      // literature and matrices are transposed, if necessary.

      double strainrateproduct = 0.0;
      double beta00;
      double beta11;
      double beta22;
      double beta01;
      double beta02;
      double beta12;
      double bbeta;
      double hk2 = pow(volume, (2.0 / 3.0));


      for (int nn = 0; nn < NSD; ++nn)
      {
        for (int rr = 0; rr < NSD; ++rr)
        {
          vderxy(nn, rr) = derxy(rr, 0) * evel(nn, 0);
          for (int mm = 1; mm < iel; ++mm)
          {
            vderxy(nn, rr) += derxy(rr, mm) * evel(nn, mm);
          }
          (*alphaij_hat)[rr][nn] = vderxy(nn, rr);
        }
      }

      // CORE::LINALG::Matrix<NSD,NSD> epsilon;
      for (int nn = 0; nn < NSD; ++nn)
      {
        for (int rr = 0; rr < NSD; ++rr)
        {
          epsilon(nn, rr) = 0.5 * (vderxy(nn, rr) + vderxy(rr, nn));
          (*strainrate_hat)[nn][rr] = epsilon(nn, rr);  // symmetric
        }
      }

      // add contribution to integral over the modeled part of subgrid
      // scale stresses
      for (int rr = 0; rr < NSD; ++rr)
      {
        for (int nn = 0; nn < NSD; ++nn)
        {
          //(*alphaij_hat)[rr][nn] *= volume;//not here, but at the end
          //(*strainrate_hat)[rr][nn] *= volume;
          alpha2_hat += (*alphaij_hat)[rr][nn] * (*alphaij_hat)[rr][nn];
          strainrateproduct += (*strainrate_hat)[rr][nn] * (*strainrate_hat)[rr][nn];
        }
      }

      beta00 = hk2 * (*alphaij_hat)[0][0] * (*alphaij_hat)[0][0] +
               hk2 * (*alphaij_hat)[1][0] * (*alphaij_hat)[1][0] +
               hk2 * (*alphaij_hat)[2][0] * (*alphaij_hat)[2][0];
      beta11 = hk2 * (*alphaij_hat)[0][1] * (*alphaij_hat)[0][1] +
               hk2 * (*alphaij_hat)[1][1] * (*alphaij_hat)[1][1] +
               hk2 * (*alphaij_hat)[2][1] * (*alphaij_hat)[2][1];
      beta22 = hk2 * (*alphaij_hat)[0][2] * (*alphaij_hat)[0][2] +
               hk2 * (*alphaij_hat)[1][2] * (*alphaij_hat)[1][2] +
               hk2 * (*alphaij_hat)[2][2] * (*alphaij_hat)[2][2];
      beta01 = hk2 * (*alphaij_hat)[0][0] * (*alphaij_hat)[0][1] +
               hk2 * (*alphaij_hat)[1][0] * (*alphaij_hat)[1][1] +
               hk2 * (*alphaij_hat)[2][0] * (*alphaij_hat)[2][1];
      beta02 = hk2 * (*alphaij_hat)[0][0] * (*alphaij_hat)[0][2] +
               hk2 * (*alphaij_hat)[1][0] * (*alphaij_hat)[1][2] +
               hk2 * (*alphaij_hat)[2][0] * (*alphaij_hat)[2][2];
      beta12 = hk2 * (*alphaij_hat)[0][1] * (*alphaij_hat)[0][2] +
               hk2 * (*alphaij_hat)[1][1] * (*alphaij_hat)[1][2] +
               hk2 * (*alphaij_hat)[2][1] * (*alphaij_hat)[2][2];

      bbeta = beta00 * beta11 - beta01 * beta01 + beta00 * beta22 - beta02 * beta02 +
              beta11 * beta22 - beta12 * beta12;
      if (alpha2_hat < 1.0e-15)
        expression_hat = 0.0;
      else
        expression_hat = sqrt(bbeta / alpha2_hat) * strainrateproduct;

      alpha2_hat *= volume;
      expression_hat *= volume;

      for (int rr = 0; rr < NSD; ++rr)
      {
        for (int nn = 0; nn < NSD; ++nn)
        {
          (*alphaij_hat)[rr][nn] *= volume;
          (*strainrate_hat)[rr][nn] *= volume;
        }
      }
    }

    return;
  }  // DRT::ELEMENTS::Fluid::f3_apply_box_filter


  /*!
    \brief compute the quantities necessary to determine Cs for the dynamic
    model.

    References are


    M. Germano, U. Piomelli, P. Moin, W.H. Cabot:
    A dynamic subgrid-scale eddy viscosity model
    (Phys. Fluids 1991)

    or

    D.K. Lilly:
    A proposed modification of the Germano subgrid-scale closure method
    (Phys. Fluids 1992)

    or
    A.E. Tejada-Martinez
    Dynamic subgrid-scale modeling for large eddy simulation of turbulent
    flows with a stabilized finite element method
    (Phd thesis, Rensselaer Polytechnic Institute, Troy, New York)


    L_ij : the tensor of resolved components of the stress tensor associated
           with the scales of motion between the resolved and the test scale

    M_ij : the tensor of modeled components of the stress tensor associated
           with the scales of motion between the resolved and the test scale
           (modulo 2Cs, a constant)



    \param evelaf                        (in )  : nodal velocities
    \param vel_hat                       (in)   : integrated convolution
                                                  with velocity
    \param reystr_hat                    (in)   : integrated convolution
                                                  with reynolds stresses
    \param modeled_stress_grid_scale_hat (in)   : integrated convolution
                                                  with subgrid scale
                                                  stress tensor model
    \param filtered_dens_vel             (in)   : integrated convolution
                                                  with density times velocity (loma only)
    \param filtered_dens                 (in)   : integrated convolution
                                                  with density (loma only)
    \param filtered_dens_strainrate     (in)   : integrated convolution
                                                  with density times rate of strain (loma only)
    \param LijMij                        (out)  : trace of product of resolved
                                                  stress tensor with modeled
                                                  stress tensor
    \param MijMij                        (out)  : trace of product of modeled
                                                  resolved stress tensor with
                                                  itself
    \param CI_                           (out)  : numerator and denominator for additional trace
    model \param center                        (out)  : element center \return void

   */
  template <int iel>
  void f3_calc_smag_const_LijMij_and_MijMij(DRT::ELEMENTS::Fluid* ele,
      DRT::ELEMENTS::FluidEleParameterStd* fldpara,
      Teuchos::RCP<Epetra_MultiVector>& col_filtered_vel,
      Teuchos::RCP<Epetra_MultiVector>& col_filtered_reynoldsstress,
      Teuchos::RCP<Epetra_MultiVector>& col_filtered_modeled_subgrid_stress,
      Teuchos::RCP<Epetra_MultiVector>& col_filtered_dens_vel,
      Teuchos::RCP<Epetra_Vector>& col_filtered_dens,
      Teuchos::RCP<Epetra_Vector>& col_filtered_dens_strainrate, double& LijMij, double& MijMij,
      double& CI_numerator, double& CI_denominator, double& xcenter, double& ycenter,
      double& zcenter)
  {
    CORE::LINALG::Matrix<3, iel> evel_hat;
    CORE::LINALG::Matrix<9, iel> ereynoldsstress_hat;
    CORE::LINALG::Matrix<9, iel> efiltered_modeled_subgrid_stress_hat;
    // loma specific quantities
    CORE::LINALG::Matrix<3, iel> edensvel_hat;
    CORE::LINALG::Matrix<1, iel> edens_hat;
    CORE::LINALG::Matrix<1, iel> edensstrainrate_hat;

    for (int nn = 0; nn < iel; ++nn)
    {
      int lid = (ele->Nodes()[nn])->LID();

      for (int dimi = 0; dimi < 3; ++dimi)
      {
        evel_hat(dimi, nn) = (*((*col_filtered_vel)(dimi)))[lid];

        for (int dimj = 0; dimj < 3; ++dimj)
        {
          int index = 3 * dimi + dimj;

          ereynoldsstress_hat(index, nn) = (*((*col_filtered_reynoldsstress)(index)))[lid];

          efiltered_modeled_subgrid_stress_hat(index, nn) =
              (*((*col_filtered_modeled_subgrid_stress)(index)))[lid];
        }
      }
    }

    if (fldpara->PhysicalType() == INPAR::FLUID::loma)
    {
      for (int nn = 0; nn < iel; ++nn)
      {
        int lid = (ele->Nodes()[nn])->LID();

        edens_hat(0, nn) = (*col_filtered_dens)[lid];
        edensstrainrate_hat(0, nn) = (*col_filtered_dens_strainrate)[lid];

        for (int dimi = 0; dimi < 3; ++dimi)
        {
          edensvel_hat(dimi, nn) = (*((*col_filtered_dens_vel)(dimi)))[lid];
        }
      }
    }

    // set element data
    const CORE::FE::CellType distype = ele->Shape();

    // allocate arrays for shapefunctions, derivatives and the transposed jacobian
    CORE::LINALG::Matrix<iel, 1> funct;
    CORE::LINALG::Matrix<3, iel> deriv;

    // this will be the elements center
    xcenter = 0.0;
    ycenter = 0.0;
    zcenter = 0.0;

    // get node coordinates of element
    CORE::LINALG::Matrix<3, iel> xyze;
    for (int inode = 0; inode < iel; inode++)
    {
      xyze(0, inode) = ele->Nodes()[inode]->X()[0];
      xyze(1, inode) = ele->Nodes()[inode]->X()[1];
      xyze(2, inode) = ele->Nodes()[inode]->X()[2];

      xcenter += xyze(0, inode);
      ycenter += xyze(1, inode);
      zcenter += xyze(2, inode);
    }
    xcenter /= iel;
    ycenter /= iel;
    zcenter /= iel;


    // use one point gauss
    CORE::FE::GaussRule3D integrationrule_filter = CORE::FE::GaussRule3D::hex_1point;
    switch (distype)
    {
      case CORE::FE::CellType::hex8:
      case CORE::FE::CellType::hex20:
      case CORE::FE::CellType::hex27:
        integrationrule_filter = CORE::FE::GaussRule3D::hex_1point;
        break;
      case CORE::FE::CellType::tet4:
      case CORE::FE::CellType::tet10:
        integrationrule_filter = CORE::FE::GaussRule3D::tet_1point;
        break;
      default:
        dserror("invalid discretization type for fluid3");
    }

    // gaussian points --- i.e. the midpoint
    const CORE::FE::IntegrationPoints3D intpoints_onepoint(integrationrule_filter);
    const double e1 = intpoints_onepoint.qxg[0][0];
    const double e2 = intpoints_onepoint.qxg[0][1];
    const double e3 = intpoints_onepoint.qxg[0][2];

    // shape functions and derivs at element center
    CORE::FE::shape_function_3D(funct, e1, e2, e3, distype);
    CORE::FE::shape_function_3D_deriv1(deriv, e1, e2, e3, distype);

    CORE::LINALG::Matrix<3, 3> xjm;
    // get Jacobian matrix and its determinant
    for (int nn = 0; nn < 3; ++nn)
    {
      for (int rr = 0; rr < 3; ++rr)
      {
        xjm(nn, rr) = deriv(nn, 0) * xyze(rr, 0);
        for (int mm = 1; mm < iel; ++mm)
        {
          xjm(nn, rr) += deriv(nn, mm) * xyze(rr, mm);
        }
      }
    }
    const double det = xjm(0, 0) * xjm(1, 1) * xjm(2, 2) + xjm(0, 1) * xjm(1, 2) * xjm(2, 0) +
                       xjm(0, 2) * xjm(1, 0) * xjm(2, 1) - xjm(0, 2) * xjm(1, 1) * xjm(2, 0) -
                       xjm(0, 0) * xjm(1, 2) * xjm(2, 1) - xjm(0, 1) * xjm(1, 0) * xjm(2, 2);

    //
    //             compute global first derivates
    //
    CORE::LINALG::Matrix<3, iel> derxy;
    /*
      Use the Jacobian and the known derivatives in element coordinate
      directions on the right hand side to compute the derivatives in
      global coordinate directions

            +-                 -+     +-    -+      +-    -+
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |     | ---- |      | ---- |
            |  dr    dr    dr   |     |  dx  |      |  dr  |
            |                   |     |      |      |      |
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |  *  | ---- |   =  | ---- | for all k
            |  ds    ds    ds   |     |  dy  |      |  ds  |
            |                   |     |      |      |      |
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |     | ---- |      | ---- |
            |  dt    dt    dt   |     |  dz  |      |  dt  |
            +-                 -+     +-    -+      +-    -+

    */
    CORE::LINALG::Matrix<3, 3> xji;
    xji(0, 0) = (xjm(1, 1) * xjm(2, 2) - xjm(2, 1) * xjm(1, 2)) / det;
    xji(1, 0) = (-xjm(1, 0) * xjm(2, 2) + xjm(2, 0) * xjm(1, 2)) / det;
    xji(2, 0) = (xjm(1, 0) * xjm(2, 1) - xjm(2, 0) * xjm(1, 1)) / det;
    xji(0, 1) = (-xjm(0, 1) * xjm(2, 2) + xjm(2, 1) * xjm(0, 2)) / det;
    xji(1, 1) = (xjm(0, 0) * xjm(2, 2) - xjm(2, 0) * xjm(0, 2)) / det;
    xji(2, 1) = (-xjm(0, 0) * xjm(2, 1) + xjm(2, 0) * xjm(0, 1)) / det;
    xji(0, 2) = (xjm(0, 1) * xjm(1, 2) - xjm(1, 1) * xjm(0, 2)) / det;
    xji(1, 2) = (-xjm(0, 0) * xjm(1, 2) + xjm(1, 0) * xjm(0, 2)) / det;
    xji(2, 2) = (xjm(0, 0) * xjm(1, 1) - xjm(1, 0) * xjm(0, 1)) / det;

    // compute global derivates
    for (int nn = 0; nn < 3; ++nn)
    {
      for (int rr = 0; rr < iel; ++rr)
      {
        derxy(nn, rr) = deriv(0, rr) * xji(nn, 0);
        for (int mm = 1; mm < 3; ++mm)
        {
          derxy(nn, rr) += deriv(mm, rr) * xji(nn, mm);
        }
      }
    }

    // get velocities (n+alpha_F/1,i) at integration point
    //
    //                   +-----
    //     ^ n+af/1       \                ^ n+af/1
    //    vel      (x) =   +      N (x) * vel
    //                     /        j         j
    //                    +-----
    //                    node j
    //
    CORE::LINALG::Matrix<3, 1> velint_hat;
    for (int rr = 0; rr < 3; ++rr)
    {
      velint_hat(rr) = funct(0) * evel_hat(rr, 0);
      for (int mm = 1; mm < iel; ++mm)
      {
        velint_hat(rr) += funct(mm) * evel_hat(rr, mm);
      }
    }

    // get velocity (n+alpha_F,i) derivatives at integration point
    //
    //     ^ n+af/1      +-----  dN (x)
    //   dvel      (x)    \        k       ^ n+af/1
    //   ------------- =   +     ------ * vel
    //       dx           /        dx        k
    //         j         +-----      j
    //                   node k
    //
    // j : direction of derivative x/y/z
    //
    CORE::LINALG::Matrix<3, 3> vderxy_hat;

    for (int nn = 0; nn < 3; ++nn)
    {
      for (int rr = 0; rr < 3; ++rr)
      {
        vderxy_hat(nn, rr) = derxy(rr, 0) * evel_hat(nn, 0);
        for (int mm = 1; mm < iel; ++mm)
        {
          vderxy_hat(nn, rr) += derxy(rr, mm) * evel_hat(nn, mm);
        }
      }
    }
    // get divergence
    double div_vel_hat = vderxy_hat(0, 0) + vderxy_hat(1, 1) + vderxy_hat(2, 2);

    // get filtered reynolds stress (n+alpha_F/1,i) at integration point
    //
    //                        +-----
    //        ^   n+af/1       \                   ^   n+af/1
    //    restress      (x) =   +      N (x) * restress
    //            ij           /        k              k, ij
    //                        +-----
    //                        node k
    //
    // restress_ij = rho*vel_i*vel_j
    // remark: this quantity is density weighted for loma
    CORE::LINALG::Matrix<3, 3> restress_hat;

    for (int nn = 0; nn < 3; ++nn)
    {
      for (int rr = 0; rr < 3; ++rr)
      {
        int index = 3 * nn + rr;
        restress_hat(nn, rr) = funct(0) * ereynoldsstress_hat(index, 0);

        for (int mm = 1; mm < iel; ++mm)
        {
          restress_hat(nn, rr) += funct(mm) * ereynoldsstress_hat(index, mm);
        }
      }
    }

    // get filtered modeled subgrid stress (n+alpha_F/1,i) at integration point
    //
    //
    //                   ^                   n+af/1
    //    filtered_modeled_subgrid_stress_hat      (x) =
    //                                       ij
    //
    //            +-----
    //             \                              ^                   n+af/1
    //          =   +      N (x) * filtered_modeled_subgrid_stress_hat
    //             /        k                                         k, ij
    //            +-----
    //            node k
    //
    // filtered_modeled_subgrid_stress_hat_ij = rho*strainrate*(epsilon_ij - 1/3*div_vel*delta_ij)
    // remark: this quantity is density weighted for loma
    CORE::LINALG::Matrix<3, 3> filtered_modeled_subgrid_stress_hat;
    for (int nn = 0; nn < 3; ++nn)
    {
      for (int rr = 0; rr < 3; ++rr)
      {
        int index = 3 * nn + rr;
        filtered_modeled_subgrid_stress_hat(nn, rr) =
            funct(0) * efiltered_modeled_subgrid_stress_hat(index, 0);

        for (int mm = 1; mm < iel; ++mm)
        {
          filtered_modeled_subgrid_stress_hat(nn, rr) +=
              funct(mm) * efiltered_modeled_subgrid_stress_hat(index, mm);
        }
      }
    }

    // get additional loma related quantities
    //----------------------------------
    CORE::LINALG::Matrix<3, 1> densvelint_hat;
    double densint_hat = 0.0;
    double densstrainrateint_hat = 0.0;

    if (fldpara->PhysicalType() == INPAR::FLUID::loma)
    {
      // get filtered density times velocity at integration point
      /*
                   +-----
            ^       \
         rho*vel =   +    N (x) * filtered_dens_vel_hat
                    /      k                           k
                   +-----
                   node k
      */
      //
      for (int rr = 0; rr < 3; ++rr)
      {
        densvelint_hat(rr) = funct(0) * edensvel_hat(rr, 0);
        for (int mm = 1; mm < iel; ++mm)
        {
          densvelint_hat(rr) += funct(mm) * edensvel_hat(rr, mm);
        }
      }

      // get filtered density at integration point
      /*
               +-----
          ^     \
         rho =   +    N (x) * filtered_dens_hat
                /      k                       k
               +-----
               node k
      */
      //
      //
      for (int mm = 0; mm < iel; ++mm)
      {
        densint_hat += funct(mm) * edens_hat(0, mm);
      }

      // get filtered density times rate of strain at integration point
      /*
                          +-----
            ^              \
         rho*strainrate =   +    N (x) * filtered_dens_strainrate
                           /      k                              k
                          +-----
                           node k
      */
      // filtered_dens_strainrate = rho*rateofstrain*rateofstrain
      //
      for (int mm = 0; mm < iel; ++mm)
      {
        densstrainrateint_hat += funct(mm) * edensstrainrate_hat(0, mm);
      }
    }
    else
    {
      // get density from material
      Teuchos::RCP<MAT::Material> material = ele->Material();
      if (material->MaterialType() == INPAR::MAT::m_fluid)
      {
        const MAT::NewtonianFluid* actmat = static_cast<const MAT::NewtonianFluid*>(material.get());
        densint_hat = actmat->Density();
      }
      else
        dserror("mat fluid expected");

      // overwrite densvelint_hat since it is used below
      for (int rr = 0; rr < 3; ++rr)
      {
        densvelint_hat(rr) = densint_hat * velint_hat(rr);
      }
    }


    // calculate strain-rate tensor of filtered field
    /*
                              +-   ^ n+af/1        ^   n+af/1    -+
        ^   / h \       1.0   |  dvel_i    (x)   dvel_j      (x)  |
       eps | u   |    = --- * |  ------------- + ---------------  |
            \   / ij    2.0   |       dx              dx          |
                              +-        j               i        -+
    */

    CORE::LINALG::Matrix<3, 3> epsilon_hat;
    for (int nn = 0; nn < 3; ++nn)
    {
      for (int rr = 0; rr < 3; ++rr)
      {
        epsilon_hat(nn, rr) = 0.5 * (vderxy_hat(nn, rr) + vderxy_hat(rr, nn));
      }
    }

    //
    // modeled part of subtestfilter scale stresses
    //
    /*    +-                                 -+ 1
          |      ^   / h \       ^   / h \    | -     ^   / h \
          | 2 * eps | u   |   * eps | u   |   | 2  * eps | u   |
          |          \   / kl        \   / kl |           \   / ij
          +-                                 -+

          |                                   |
          +-----------------------------------+
               'resolved' rate of strain
    */

    double rateofstrain_hat = 0.0;

    for (int rr = 0; rr < 3; rr++)
    {
      for (int mm = 0; mm < 3; mm++)
      {
        rateofstrain_hat += epsilon_hat(rr, mm) * epsilon_hat(rr, mm);
      }
    }
    rateofstrain_hat *= 2.0;
    rateofstrain_hat = sqrt(rateofstrain_hat);

    CORE::LINALG::Matrix<3, 3> L_ij;
    CORE::LINALG::Matrix<3, 3> M_ij;

    for (int rr = 0; rr < 3; rr++)
    {
      for (int mm = 0; mm < 3; mm++)
      {
        //      L_ij(rr,mm) = restress_hat(rr,mm) - velint_hat(rr)*velint_hat(mm);
        L_ij(rr, mm) = restress_hat(rr, mm) - densvelint_hat(rr) * densvelint_hat(mm) / densint_hat;
        // this part can be neglected, since contraction with deviatoric tensor M cancels it out
        // see, e.g., phd thesis peter gamnitzer (der gammi!)
        //      if (rr==mm)
        //          L_ij(rr,mm) -= 1.0/3.0 * ((restress_hat(0,0) -
        //          densvelint_hat(0)*densvelint_hat(0)/densint_hat)
        //                                   +(restress_hat(1,1) -
        //                                   densvelint_hat(1)*densvelint_hat(1)/densint_hat)
        //                                   +(restress_hat(2,2) -
        //                                   densvelint_hat(2)*densvelint_hat(2)/densint_hat));
      }
    }

    // this is sqrt(3)
    const double filterwidthratio = 1.73;

    for (int rr = 0; rr < 3; rr++)
    {
      for (int mm = 0; mm < 3; mm++)
      {
        //      M_ij(rr,mm) = filtered_modeled_subgrid_stress_hat(rr,mm)
        //        -
        //        filterwidthratio*filterwidthratio*rateofstrain_hat*epsilon_hat(rr,mm);
        M_ij(rr, mm) = filtered_modeled_subgrid_stress_hat(rr, mm) -
                       filterwidthratio * filterwidthratio * densint_hat * rateofstrain_hat *
                           epsilon_hat(rr, mm);
        if (fldpara->PhysicalType() == INPAR::FLUID::loma and rr == mm)
          M_ij(rr, mm) += filterwidthratio * filterwidthratio * densint_hat * rateofstrain_hat *
                          1.0 / 3.0 * div_vel_hat;
      }
    }

    LijMij = 0.0;
    MijMij = 0.0;
    for (int rr = 0; rr < 3; rr++)
    {
      for (int mm = 0; mm < 3; mm++)
      {
        LijMij += L_ij(rr, mm) * M_ij(rr, mm);
        MijMij += M_ij(rr, mm) * M_ij(rr, mm);
      }
    }

    // calculate CI for trace of modeled subgrid-stress tensor (loma only)
    if (fldpara->PhysicalType() == INPAR::FLUID::loma)
    {
      CI_numerator = restress_hat(0, 0) + restress_hat(1, 1) + restress_hat(2, 2) -
                     densvelint_hat.Dot(densvelint_hat) / densint_hat;
      CI_denominator =
          densint_hat * filterwidthratio * filterwidthratio * rateofstrain_hat * rateofstrain_hat -
          densstrainrateint_hat;
    }

    return;
  }  // DRT::ELEMENTS::Fluid::f3_calc_smag_const_LijMij_and_MijMij


  template <int iel>
  void f3_calc_vreman_const(DRT::ELEMENTS::Fluid* ele,
      Teuchos::RCP<Epetra_MultiVector>& col_filtered_strainrate,
      Teuchos::RCP<Epetra_MultiVector>& col_filtered_alphaij,
      Teuchos::RCP<Epetra_Vector>& col_filtered_expression,
      Teuchos::RCP<Epetra_Vector>& col_filtered_alpha2, double& cv_numerator,
      double& cv_denominator, double& volume)
  {
    CORE::LINALG::Matrix<9, iel> estrainrate_hat(true);
    CORE::LINALG::Matrix<9, iel> ealphaij_hat(true);
    CORE::LINALG::Matrix<1, iel> eexpression_hat(true);
    CORE::LINALG::Matrix<1, iel> ealpha2_hat(true);
    CORE::LINALG::Matrix<3, 3> strainrate_hat(true);
    CORE::LINALG::Matrix<3, 3> alphaij_hat(true);
    double alpha2_hat = 0.0;
    double expression_hat = 0.0;

    // get to element
    for (int nn = 0; nn < iel; ++nn)
    {
      int lid = (ele->Nodes()[nn])->LID();

      for (int dimi = 0; dimi < 3; ++dimi)
      {
        for (int dimj = 0; dimj < 3; ++dimj)
        {
          int index = 3 * dimi + dimj;

          estrainrate_hat(index, nn) = (*((*col_filtered_strainrate)(index)))[lid];

          ealphaij_hat(index, nn) = (*((*col_filtered_alphaij)(index)))[lid];
        }
      }
    }

    for (int nn = 0; nn < iel; ++nn)
    {
      int lid = (ele->Nodes()[nn])->LID();

      eexpression_hat(0, nn) = (*col_filtered_expression)[lid];
      ealpha2_hat(0, nn) = (*col_filtered_alpha2)[lid];
    }

    // number of spatial dimensions is always 3
    const int NSD = 3;


    // set element data
    const CORE::FE::CellType distype = ele->Shape();
    CORE::LINALG::Matrix<iel, 1> funct(true);
    // allocate arrays for shape functions, derivatives and the transposed jacobian
    CORE::LINALG::Matrix<NSD, NSD> xjm(true);
    CORE::LINALG::Matrix<NSD, iel> deriv(true);



    // get node coordinates of element
    CORE::LINALG::Matrix<NSD, iel> xyze(true);
    for (int inode = 0; inode < iel; inode++)
    {
      xyze(0, inode) = ele->Nodes()[inode]->X()[0];
      xyze(1, inode) = ele->Nodes()[inode]->X()[1];
      xyze(2, inode) = ele->Nodes()[inode]->X()[2];
    }

    // get gauss rule: we use a one-point rule here
    CORE::FE::GaussRule3D integrationrule_filter = CORE::FE::GaussRule3D::hex_1point;
    switch (distype)
    {
      case CORE::FE::CellType::hex8:
        integrationrule_filter = CORE::FE::GaussRule3D::hex_1point;
        break;
      case CORE::FE::CellType::tet4:
        integrationrule_filter = CORE::FE::GaussRule3D::tet_1point;
        break;
      case CORE::FE::CellType::tet10:
      case CORE::FE::CellType::hex20:
      case CORE::FE::CellType::hex27:
        dserror("the box filtering operation is only permitted for linear elements\n");
        break;
      default:
        dserror("invalid discretization type for fluid3");
    }

    // gaussian points
    const CORE::FE::IntegrationPoints3D intpoints_onepoint(integrationrule_filter);

    // shape functions and derivs at element center
    const double e1 = intpoints_onepoint.qxg[0][0];
    const double e2 = intpoints_onepoint.qxg[0][1];
    const double e3 = intpoints_onepoint.qxg[0][2];
    const double wquad = intpoints_onepoint.qwgt[0];

    CORE::FE::shape_function_3D(funct, e1, e2, e3, distype);
    CORE::FE::shape_function_3D_deriv1(deriv, e1, e2, e3, distype);

    for (int nn = 0; nn < 3; ++nn)
    {
      for (int rr = 0; rr < 3; ++rr)
      {
        int index = 3 * nn + rr;
        strainrate_hat(nn, rr) = funct(0) * estrainrate_hat(index, 0);
        alphaij_hat(nn, rr) = funct(0) * ealphaij_hat(index, 0);
        for (int mm = 1; mm < iel; ++mm)
        {
          strainrate_hat(nn, rr) += funct(mm) * estrainrate_hat(index, mm);
          alphaij_hat(nn, rr) += funct(mm) * ealphaij_hat(index, mm);
        }
      }
    }

    for (int mm = 0; mm < iel; ++mm)
    {
      alpha2_hat += funct(mm) * ealpha2_hat(0, mm);
      expression_hat += funct(mm) * eexpression_hat(0, mm);
    }
    // get Jacobian matrix and determinant
    for (int nn = 0; nn < NSD; ++nn)
    {
      for (int rr = 0; rr < NSD; ++rr)
      {
        xjm(nn, rr) = deriv(nn, 0) * xyze(rr, 0);
        for (int mm = 1; mm < iel; ++mm)
        {
          xjm(nn, rr) += deriv(nn, mm) * xyze(rr, mm);
        }
      }
    }

    const double det = xjm(0, 0) * xjm(1, 1) * xjm(2, 2) + xjm(0, 1) * xjm(1, 2) * xjm(2, 0) +
                       xjm(0, 2) * xjm(1, 0) * xjm(2, 1) - xjm(0, 2) * xjm(1, 1) * xjm(2, 0) -
                       xjm(0, 0) * xjm(1, 2) * xjm(2, 1) - xjm(0, 1) * xjm(1, 0) * xjm(2, 2);


    // In the literature about the Vreman model, the indices i and j are swapped compared to the
    // standard definition in baci. All variables used for the Vreman model are used as in
    // literature and matrices are transposed, if necessary.
    volume = wquad * det;
    // calculate nominator and denominator
    {
      double beta00;
      double beta11;
      double beta22;
      double beta01;
      double beta02;
      double beta12;
      double bbeta;
      double hk2 =
          3.0 *
          pow(volume, (2.0 / 3.0));  // times 3 because the filter width of the box filter is
                                     // assumed to be sqrt(3) larger than the implicit grid filter
      double PIg = 0.0;
      double alphavreman = 0.0;
      double STRAINRATE = 0.0;


      alphavreman = alphaij_hat(0, 0) * alphaij_hat(0, 0) + alphaij_hat(0, 1) * alphaij_hat(0, 1) +
                    alphaij_hat(0, 2) * alphaij_hat(0, 2) + alphaij_hat(1, 0) * alphaij_hat(1, 0) +
                    alphaij_hat(1, 1) * alphaij_hat(1, 1) + alphaij_hat(1, 2) * alphaij_hat(1, 2) +
                    alphaij_hat(2, 0) * alphaij_hat(2, 0) + alphaij_hat(2, 1) * alphaij_hat(2, 1) +
                    alphaij_hat(2, 2) * alphaij_hat(2, 2);

      STRAINRATE = strainrate_hat(0, 0) * strainrate_hat(0, 0) +
                   strainrate_hat(0, 1) * strainrate_hat(0, 1) +
                   strainrate_hat(0, 2) * strainrate_hat(0, 2) +
                   strainrate_hat(1, 0) * strainrate_hat(1, 0) +
                   strainrate_hat(1, 1) * strainrate_hat(1, 1) +
                   strainrate_hat(1, 2) * strainrate_hat(1, 2) +
                   strainrate_hat(2, 0) * strainrate_hat(2, 0) +
                   strainrate_hat(2, 1) * strainrate_hat(2, 1) +
                   strainrate_hat(2, 2) * strainrate_hat(2, 2);

      beta00 = hk2 * alphaij_hat(0, 0) * alphaij_hat(0, 0) +
               hk2 * alphaij_hat(1, 0) * alphaij_hat(1, 0) +
               hk2 * alphaij_hat(2, 0) * alphaij_hat(2, 0);
      beta11 = hk2 * alphaij_hat(0, 1) * alphaij_hat(0, 1) +
               hk2 * alphaij_hat(1, 1) * alphaij_hat(1, 1) +
               hk2 * alphaij_hat(2, 1) * alphaij_hat(2, 1);
      beta22 = hk2 * alphaij_hat(0, 2) * alphaij_hat(0, 2) +
               hk2 * alphaij_hat(1, 2) * alphaij_hat(1, 2) +
               hk2 * alphaij_hat(2, 2) * alphaij_hat(2, 2);
      beta01 = hk2 * alphaij_hat(0, 0) * alphaij_hat(0, 1) +
               hk2 * alphaij_hat(1, 0) * alphaij_hat(1, 1) +
               hk2 * alphaij_hat(2, 0) * alphaij_hat(2, 1);
      beta02 = hk2 * alphaij_hat(0, 0) * alphaij_hat(0, 2) +
               hk2 * alphaij_hat(1, 0) * alphaij_hat(1, 2) +
               hk2 * alphaij_hat(2, 0) * alphaij_hat(2, 2);
      beta12 = hk2 * alphaij_hat(0, 1) * alphaij_hat(0, 2) +
               hk2 * alphaij_hat(1, 1) * alphaij_hat(1, 2) +
               hk2 * alphaij_hat(2, 1) * alphaij_hat(2, 2);

      bbeta = beta00 * beta11 - beta01 * beta01 + beta00 * beta22 - beta02 * beta02 +
              beta11 * beta22 - beta12 * beta12;
      if (alphavreman < 1.0e-15)
        PIg = 0.0;
      else
        PIg = sqrt(bbeta / alphavreman);
      cv_numerator = volume * (alpha2_hat - alphavreman);
      cv_denominator = volume * (expression_hat - PIg * STRAINRATE);
    }

    return;
  }  // DRT::ELEMENTS::Fluid::f3_calc_vreman_const


  /*!
   \brief compute parameters of multifractal subgrid-scale model
  */
  template <int NEN, int NSD, CORE::FE::CellType DISTYPE>
  void f3_get_mf_params(DRT::ELEMENTS::Fluid* ele, DRT::ELEMENTS::FluidEleParameterStd* fldpara,
      Teuchos::ParameterList& params, Teuchos::RCP<MAT::Material> mat, std::vector<double>& vel,
      std::vector<double>& fsvel)
  {
    // get mfs parameter
    Teuchos::ParameterList* turbmodelparamsmfs = &(params.sublist("MULTIFRACTAL SUBGRID SCALES"));
    bool withscatra = params.get<bool>("scalar");

    // allocate a fixed size array for nodal velocities
    CORE::LINALG::Matrix<NSD, NEN> evel;
    CORE::LINALG::Matrix<NSD, NEN> efsvel;

    // split velocity and throw away  pressure, insert into element array
    for (int inode = 0; inode < NEN; inode++)
    {
      for (int idim = 0; idim < NSD; idim++)
      {
        evel(idim, inode) = vel[inode * 4 + idim];
        efsvel(idim, inode) = fsvel[inode * 4 + idim];
      }
    }

    // get material
    double dynvisc = 0.0;
    double dens = 0.0;
    if (mat->MaterialType() == INPAR::MAT::m_fluid)
    {
      const MAT::NewtonianFluid* actmat = static_cast<const MAT::NewtonianFluid*>(mat.get());
      // get constant viscosity
      dynvisc = actmat->Viscosity();
      // get constant density
      dens = actmat->Density();
      if (dens == 0.0 or dynvisc == 0.0)
      {
        dserror("Could not get material parameters!");
      }
    }

    // allocate array for gauss-point velocities and derivatives
    CORE::LINALG::Matrix<NSD, 1> velint;
    CORE::LINALG::Matrix<NSD, NSD> velintderxy;
    CORE::LINALG::Matrix<NSD, 1> fsvelint;
    CORE::LINALG::Matrix<NSD, NSD> fsvelintderxy;

    // allocate arrays for shape functions and derivatives
    CORE::LINALG::Matrix<NEN, 1> funct;
    CORE::LINALG::Matrix<NSD, NEN> deriv;
    CORE::LINALG::Matrix<NSD, NEN> derxy;
    double vol = 0.0;

    // array for element coordinates in physical space
    CORE::LINALG::Matrix<NSD, NEN> xyze;
    // this will be the y-coordinate of the element center
    double center = 0;
    // get node coordinates of element
    for (int inode = 0; inode < NEN; inode++)
    {
      for (int idim = 0; idim < NSD; idim++) xyze(idim, inode) = ele->Nodes()[inode]->X()[idim];

      center += xyze(1, inode);
    }
    center /= NEN;

    // evaluate shape functions and derivatives at element center
    CORE::LINALG::Matrix<NSD, NSD> xji;
    {
      // use one-point Gauss rule
      CORE::FE::IntPointsAndWeights<NSD> intpoints(
          DRT::ELEMENTS::DisTypeToStabGaussRule<DISTYPE>::rule);

      // coordinates of the current integration point
      const double* gpcoord = (intpoints.IP().qxg)[0];
      CORE::LINALG::Matrix<NSD, 1> xsi;
      for (int idim = 0; idim < NSD; idim++)
      {
        xsi(idim) = gpcoord[idim];
      }
      const double wquad = intpoints.IP().qwgt[0];

      // shape functions and their first derivatives
      CORE::FE::shape_function<DISTYPE>(xsi, funct);
      CORE::FE::shape_function_deriv1<DISTYPE>(xsi, deriv);

      // get Jacobian matrix and determinant
      CORE::LINALG::Matrix<NSD, NSD> xjm;
      // CORE::LINALG::Matrix<NSD,NSD> xji;
      xjm.MultiplyNT(deriv, xyze);
      double det = xji.Invert(xjm);
      // check for degenerated elements
      if (det < 1E-16)
        dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", ele->Id(), det);

      // set element area or volume
      vol = wquad * det;

      // compute global first derivatives
      derxy.Multiply(xji, deriv);
    }

    // calculate parameters of multifractal subgrid-scales
    // set input parameters
    double Csgs = turbmodelparamsmfs->get<double>("CSGS");
    double alpha = 0.0;
    if (turbmodelparamsmfs->get<std::string>("SCALE_SEPARATION") == "algebraic_multigrid_operator")
      alpha = 3.0;
    else if (turbmodelparamsmfs->get<std::string>("SCALE_SEPARATION") == "box_filter")
      alpha = 2.0;
    else
      dserror("Unknown filter type!");
    // allocate vector for parameter N
    // N may depend on the direction
    std::vector<double> Nvel(3);
    // element Reynolds number
    double Re_ele = -1.0;
    // characteristic element length
    double hk = 1.0e+10;

    // calculate norm of strain rate
    double strainnorm = 0.0;
    // compute (resolved) norm of strain
    //
    //          +-                                 -+ 1
    //          |          /   \           /   \    | -
    //          |     eps | vel |   * eps | vel |   | 2
    //          |          \   / ij        \   / ij |
    //          +-                                 -+
    //
    velintderxy.MultiplyNT(evel, derxy);
    CORE::LINALG::Matrix<NSD, NSD> twoeps;
    for (int idim = 0; idim < NSD; idim++)
    {
      for (int jdim = 0; jdim < NSD; jdim++)
      {
        twoeps(idim, jdim) = velintderxy(idim, jdim) + velintderxy(jdim, idim);
      }
    }

    for (int idim = 0; idim < NSD; idim++)
    {
      for (int jdim = 0; jdim < NSD; jdim++)
      {
        strainnorm += twoeps(idim, jdim) * twoeps(idim, jdim);
      }
    }
    strainnorm = (sqrt(strainnorm / 4.0));

    // do we have a fixed parameter N
    if ((CORE::UTILS::IntegralValue<int>(*turbmodelparamsmfs, "CALC_N")) == false)
    {
      for (int rr = 1; rr < 3; rr++) Nvel[rr] = turbmodelparamsmfs->get<double>("N");
    }
    else  // no, so we calculate N from Re
    {
      double scale_ratio = 0.0;

      // get velocity at element center
      velint.Multiply(evel, funct);
      fsvelint.Multiply(efsvel, funct);
      // get norm
      const double vel_norm = velint.Norm2();
      const double fsvel_norm = fsvelint.Norm2();

      // calculate characteristic element length
      // cf. stabilization parameters
      INPAR::FLUID::RefLength reflength = INPAR::FLUID::cube_edge;
      if (turbmodelparamsmfs->get<std::string>("REF_LENGTH") == "cube_edge")
        reflength = INPAR::FLUID::cube_edge;
      else if (turbmodelparamsmfs->get<std::string>("REF_LENGTH") == "sphere_diameter")
        reflength = INPAR::FLUID::sphere_diameter;
      else if (turbmodelparamsmfs->get<std::string>("REF_LENGTH") == "streamlength")
        reflength = INPAR::FLUID::streamlength;
      else if (turbmodelparamsmfs->get<std::string>("REF_LENGTH") == "gradient_based")
        reflength = INPAR::FLUID::gradient_based;
      else if (turbmodelparamsmfs->get<std::string>("REF_LENGTH") == "metric_tensor")
        reflength = INPAR::FLUID::metric_tensor;
      else
        dserror("Unknown length!");
      switch (reflength)
      {
        case INPAR::FLUID::streamlength:
        {
          // a) streamlength due to Tezduyar et al. (1992)
          // normed velocity vector
          CORE::LINALG::Matrix<NSD, 1> velino(true);
          if (vel_norm >= 1e-6)
            velino.Update(1.0 / vel_norm, velint);
          else
          {
            velino.Clear();
            velino(0, 0) = 1.0;
          }
          CORE::LINALG::Matrix<NEN, 1> tmp;
          tmp.MultiplyTN(derxy, velino);
          const double val = tmp.Norm1();
          hk = 2.0 / val;

          break;
        }
        case INPAR::FLUID::sphere_diameter:
        {
          // b) volume-equivalent diameter
          hk = std::pow((6. * vol / M_PI), (1.0 / 3.0)) / sqrt(3.0);

          break;
        }
        case INPAR::FLUID::cube_edge:
        {
          // c) qubic element length
          hk = std::pow(vol, (1.0 / (double(NSD))));
          break;
        }
        case INPAR::FLUID::metric_tensor:
        {
          /*          +-           -+   +-           -+   +-           -+
                      |             |   |             |   |             |
                      |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
                G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
                 ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                      |    i     j  |   |    i     j  |   |    i     j  |
                      +-           -+   +-           -+   +-           -+
          */
          CORE::LINALG::Matrix<3, 3> G;

          for (int nn = 0; nn < 3; ++nn)
          {
            for (int rr = 0; rr < 3; ++rr)
            {
              G(nn, rr) = xji(nn, 0) * xji(rr, 0);
              for (int mm = 1; mm < 3; ++mm)
              {
                G(nn, rr) += xji(nn, mm) * xji(rr, mm);
              }
            }
          }

          /*          +----
                       \
              G : G =   +   G   * G
              -   -    /     ij    ij
              -   -   +----
                       i,j
          */
          double normG = 0;
          for (int nn = 0; nn < 3; ++nn)
          {
            for (int rr = 0; rr < 3; ++rr)
            {
              normG += G(nn, rr) * G(nn, rr);
            }
          }
          hk = std::pow(normG, -0.25);

          break;
        }
        case INPAR::FLUID::gradient_based:
        {
          velintderxy.MultiplyNT(evel, derxy);
          CORE::LINALG::Matrix<3, 1> normed_velgrad;

          for (int rr = 0; rr < 3; ++rr)
          {
            normed_velgrad(rr) = sqrt(velintderxy(0, rr) * velintderxy(0, rr) +
                                      velintderxy(1, rr) * velintderxy(1, rr) +
                                      velintderxy(2, rr) * velintderxy(2, rr));
          }
          double norm = normed_velgrad.Norm2();

          // normed gradient
          if (norm > 1e-6)
          {
            for (int rr = 0; rr < 3; ++rr)
            {
              normed_velgrad(rr) /= norm;
            }
          }
          else
          {
            normed_velgrad(0) = 1.;
            for (int rr = 1; rr < 3; ++rr)
            {
              normed_velgrad(rr) = 0.0;
            }
          }

          // get length in this direction
          double val = 0.0;
          for (int rr = 0; rr < NEN; ++rr) /* loop element nodes */
          {
            val += fabs(normed_velgrad(0) * derxy(0, rr) + normed_velgrad(1) * derxy(1, rr) +
                        normed_velgrad(2) * derxy(2, rr));
          } /* end of loop over element nodes */

          hk = 2.0 / val;

          break;
        }
        default:
          dserror("Unknown length");
      }

        // alternative length for comparison, currently not used
#ifdef HMIN
      double xmin = 0.0;
      double ymin = 0.0;
      double zmin = 0.0;
      double xmax = 0.0;
      double ymax = 0.0;
      double zmax = 0.0;
      for (int inen = 0; inen < NEN; inen++)
      {
        if (inen == 0)
        {
          xmin = xyze(0, inen);
          xmax = xyze(0, inen);
          ymin = xyze(1, inen);
          ymax = xyze(1, inen);
          zmin = xyze(2, inen);
          zmax = xyze(2, inen);
        }
        else
        {
          if (xyze(0, inen) < xmin) xmin = xyze(0, inen);
          if (xyze(0, inen) > xmax) xmax = xyze(0, inen);
          if (xyze(1, inen) < ymin) ymin = xyze(1, inen);
          if (xyze(1, inen) > ymax) ymax = xyze(1, inen);
          if (xyze(2, inen) < zmin) zmin = xyze(2, inen);
          if (xyze(2, inen) > zmax) zmax = xyze(2, inen);
        }
      }
      if ((xmax - xmin) < (ymax - ymin))
      {
        if ((xmax - xmin) < (zmax - zmin)) hk = xmax - xmin;
      }
      else
      {
        if ((ymax - ymin) < (zmax - zmin))
          hk = ymax - ymin;
        else
          hk = zmax - zmin;
      }
#endif
#ifdef HMAX
      double xmin = 0.0;
      double ymin = 0.0;
      double zmin = 0.0;
      double xmax = 0.0;
      double ymax = 0.0;
      double zmax = 0.0;
      for (int inen = 0; inen < NEN; inen++)
      {
        if (inen == 0)
        {
          xmin = xyze(0, inen);
          xmax = xyze(0, inen);
          ymin = xyze(1, inen);
          ymax = xyze(1, inen);
          zmin = xyze(2, inen);
          zmax = xyze(2, inen);
        }
        else
        {
          if (xyze(0, inen) < xmin) xmin = xyze(0, inen);
          if (xyze(0, inen) > xmax) xmax = xyze(0, inen);
          if (xyze(1, inen) < ymin) ymin = xyze(1, inen);
          if (xyze(1, inen) > ymax) ymax = xyze(1, inen);
          if (xyze(2, inen) < zmin) zmin = xyze(2, inen);
          if (xyze(2, inen) > zmax) zmax = xyze(2, inen);
        }
      }
      if ((xmax - xmin) > (ymax - ymin))
      {
        if ((xmax - xmin) > (zmax - zmin)) hk = xmax - xmin;
      }
      else
      {
        if ((ymax - ymin) > (zmax - zmin))
          hk = ymax - ymin;
        else
          hk = zmax - zmin;
      }
#endif

      if (hk == 1.0e+10) dserror("Something went wrong!");

      // get reference velocity
      INPAR::FLUID::RefVelocity refvel = INPAR::FLUID::strainrate;
      if (turbmodelparamsmfs->get<std::string>("REF_VELOCITY") == "strainrate")
        refvel = INPAR::FLUID::strainrate;
      else if (turbmodelparamsmfs->get<std::string>("REF_VELOCITY") == "resolved")
        refvel = INPAR::FLUID::resolved;
      else if (turbmodelparamsmfs->get<std::string>("REF_VELOCITY") == "fine_scale")
        refvel = INPAR::FLUID::fine_scale;
      else
        dserror("Unknown velocity!");

      switch (refvel)
      {
        case INPAR::FLUID::resolved:
        {
          Re_ele = vel_norm * hk * dens / dynvisc;
          break;
        }
        case INPAR::FLUID::fine_scale:
        {
          Re_ele = fsvel_norm * hk * dens / dynvisc;
          break;
        }
        case INPAR::FLUID::strainrate:
        {
          Re_ele = strainnorm * hk * hk * dens / dynvisc;
          break;
        }
        default:
          dserror("Unknown velocity!");
      }
      if (Re_ele < 0.0) dserror("Something went wrong!");

      if (Re_ele < 1.0) Re_ele = 1.0;

      //
      //   Delta
      //  ---------  ~ Re^(3/4)
      //  lambda_nu
      //
      scale_ratio = turbmodelparamsmfs->get<double>("C_NU") * pow(Re_ele, 0.75);
      // scale_ration < 1.0 leads to N < 0
      // therefore, we clip once more
      if (scale_ratio < 1.0) scale_ratio = 1.0;

      //         |   Delta     |
      //  N =log | ----------- |
      //        2|  lambda_nu  |
      double N_re = log(scale_ratio) / log(2.0);
      if (N_re < 0.0) dserror("Something went wrong when calculating N!");

      for (int i = 0; i < NSD; i++) Nvel[i] = N_re;
    }


    // calculate coefficient of subgrid-velocity
    // allocate array for coefficient B
    // B may depend on the direction (if N depends on it)
    CORE::LINALG::Matrix<NSD, 1> B(true);
    //                                  1
    //          |       1              |2
    //  kappa = | -------------------- |
    //          |  1 - alpha ^ (-4/3)  |
    //
    double kappa = 1.0 / (1.0 - pow(alpha, -4.0 / 3.0));

    //                                                     1
    //                                  |                 |2
    //  B = CI * kappa * 2 ^ (-2*N/3) * | 2 ^ (4*N/3) - 1 |
    //                                  |                 |
    //


    // calculate near-wall correction
    double Cai_phi = 0.0;
    if ((CORE::UTILS::IntegralValue<int>(*turbmodelparamsmfs, "NEAR_WALL_LIMIT")) == true)
    {
      // get Re from strain rate
      double Re_ele_str = strainnorm * hk * hk * dens / dynvisc;
      if (Re_ele_str < 0.0) dserror("Something went wrong!");
      // ensure positive values
      if (Re_ele_str < 1.0) Re_ele_str = 1.0;

      // calculate corrected Csgs
      //           -3/16
      //  *(1 - (Re)   )
      //
      Csgs *= (1.0 - pow(Re_ele_str, -3.0 / 16.0));

      // store Cai for application to scalar field
      Cai_phi = (1.0 - pow(Re_ele_str, -3.0 / 16.0));
    }

    for (int dim = 0; dim < NSD; dim++)
    {
      B(dim, 0) = Csgs * sqrt(kappa) * pow(2.0, -2.0 * Nvel[0] / 3.0) *
                  sqrt((pow(2.0, 4.0 * Nvel[0] / 3.0) - 1.0));
    }

    // calculate model parameters for passive scalar transport
    // allocate vector for parameter N
    // N may depend on the direction -> currently unused
    double Nphi = 0.0;
    // allocate array for coefficient D
    // D may depend on the direction (if N depends on it)
    double D = 0.0;
    double Csgs_phi = fldpara->CsgsPhi();  // turbmodelparamsmfs->get<double>("CSGS_PHI");

    if (withscatra)
    {
      // get Schmidt number
      double scnum = params.get<double>("scnum");
      // ratio of dissipation scale to element length
      double scale_ratio_phi = 0.0;

      if ((CORE::UTILS::IntegralValue<int>(*turbmodelparamsmfs, "CALC_N")) == true)
      {
        //
        //   Delta
        //  ---------  ~ Re^(3/4)*Sc^(p)
        //  lambda_diff
        //
        // Sc <= 1: p=3/4
        // Sc >> 1: p=1/2
        double p = 0.75;
        if (scnum > 1.0) p = 0.5;

        scale_ratio_phi =
            turbmodelparamsmfs->get<double>("C_DIFF") * pow(Re_ele, 0.75) * pow(scnum, p);
        // scale_ratio < 1.0 leads to N < 0
        // therefore, we clip again
        if (scale_ratio_phi < 1.0) scale_ratio_phi = 1.0;

        //         |   Delta     |
        //  N =log | ----------- |
        //        2|  lambda_nu  |
        Nphi = log(scale_ratio_phi) / log(2.0);
        if (Nphi < 0.0) dserror("Something went wrong when calculating N!");
      }
      else
        dserror("Multifractal subgrid-scales for scalar transport with calculation of N, only!");

      // here, we have to distinguish three different cases:
      // Sc ~ 1 : fluid and scalar field have the nearly the same cutoff (usual case)
      //          k^(-5/3) scaling -> gamma = 4/3
      // Sc >> 1: (i)  cutoff in the inertial-convective range (Nvel>0, tricky!)
      //               k^(-5/3) scaling in the inertial-convective range
      //               k^(-1) scaling in the viscous-convective range
      //          (ii) cutoff in the viscous-convective range (fluid field fully resolved, easier)
      //               k^(-1) scaling -> gamma = 2
      // rare:
      // Sc << 1: fluid field could be fully resolved, not necessary
      //          k^(-5/3) scaling -> gamma = 4/3
      // Remark: case 2.(i) not implemented, yet

      double gamma = 0.0;
      // define limit to distinguish between low and high Schmidt number
      const double Sc_limit = 2.0;
      // special option for case 2 (i)
      bool two_ranges = false;
      if (scnum < Sc_limit)  // Sc <= 1, i.e., case 1 and 3
        gamma = 4.0 / 3.0;
      else  // Pr >> 1
      {
        if (fldpara->PhysicalType() == INPAR::FLUID::loma) dserror("Loma with Pr>>1?");
        if (Nvel[0] < 1.0)  // Sc >> 1 and fluid fully resolved, i.e., case 2 (ii)
          gamma = 2.0;
        else  // Sc >> 1 and fluid not fully resolved, i.e., case 2 (i)
        {
          if (Nvel[0] > Nphi) dserror("Nvel < Nphi expected!");
          // here different options are possible
          // 1) we assume k^(-5/3) for the complete range
          gamma = 4.0 / 3.0;
        }
      }

      //
      //   Phi    |       1                |
      //  kappa = | ---------------------- |
      //          |  1 - alpha ^ (-gamma)  |
      //
      double kappa_phi = 1.0 / (1.0 - pow(alpha, -gamma));

      // calculate coefficient of subgrid-scalar
      //                                                             1
      //       Phi    Phi                       |                   |2
      //  D = Csgs * kappa * 2 ^ (-gamma*N/2) * | 2 ^ (gamma*N) - 1 |
      //                                        |                   |
      //
      if (not two_ranges)  // usual case
        D = Csgs_phi * sqrt(kappa_phi) * pow(2.0, -gamma * Nphi / 2.0) *
            sqrt((pow(2.0, gamma * Nphi) - 1.0));
      else
      {
        double gamma1 = 4.0 / 3.0;
        double gamma2 = 2.0;
        kappa_phi = 1.0 / (1.0 - pow(alpha, -gamma1));
        D = Csgs_phi * sqrt(kappa_phi) * pow(2.0, -gamma2 * Nphi / 2.0) *
            sqrt((pow(2.0, gamma1 * Nvel[0]) - 1) +
                 4.0 / 3.0 * (M_PI / hk) * (pow(2.0, gamma2 * Nphi) - pow(2.0, gamma2 * Nvel[0])));
      }

      // apply near-wall limit if required
      if (((CORE::UTILS::IntegralValue<int>(*turbmodelparamsmfs, "NEAR_WALL_LIMIT_CSGS_PHI")) ==
              true) and
          ((CORE::UTILS::IntegralValue<int>(*turbmodelparamsmfs, "NEAR_WALL_LIMIT")) == true))
      {
        D *= Cai_phi;
        Csgs_phi *= Cai_phi;
      }
    }

    // calculate subgrid-viscosity, if small-scale eddy-viscosity term is included
    double sgvisc = 0.0;
    if (params.sublist("TURBULENCE MODEL").get<std::string>("FSSUGRVISC", "No") != "No")
    {
      // get filter width and Smagorinsky-coefficient
      const double hk_sgvisc = std::pow(vol, (1.0 / NSD));
      const double Cs = params.sublist("SUBGRID VISCOSITY").get<double>("C_SMAGORINSKY");

      // compute rate of strain
      //
      //          +-                                 -+ 1
      //          |          /   \           /   \    | -
      //          | 2 * eps | vel |   * eps | vel |   | 2
      //          |          \   / ij        \   / ij |
      //          +-                                 -+
      //
      CORE::LINALG::Matrix<NSD, NSD> velderxy(true);
      velintderxy.MultiplyNT(evel, derxy);
      fsvelintderxy.MultiplyNT(efsvel, derxy);

      if (params.sublist("TURBULENCE MODEL").get<std::string>("FSSUGRVISC", "No") ==
          "Smagorinsky_all")
        velderxy = velintderxy;
      else if (params.sublist("TURBULENCE MODEL").get<std::string>("FSSUGRVISC", "No") ==
               "Smagorinsky_small")
        velderxy = fsvelintderxy;
      else
        dserror("fssgvisc-type unknown");

#ifdef SUBGRID_SCALE  // unused
      for (int idim = 0; idim < NSD; idim++)
      {
        for (int jdim = 0; jdim < NSD; jdim++)
          mffsvelintderxy_(idim, jdim) = fsvelintderxy_(idim, jdim) * B(idim, 0);
      }
      velderxy = mffsvelintderxy;
#endif

      CORE::LINALG::Matrix<NSD, NSD> two_epsilon;
      double rateofstrain = 0.0;
      for (int idim = 0; idim < NSD; idim++)
      {
        for (int jdim = 0; jdim < NSD; jdim++)
        {
          two_epsilon(idim, jdim) = velderxy(idim, jdim) + velderxy(jdim, idim);
        }
      }

      for (int idim = 0; idim < NSD; idim++)
      {
        for (int jdim = 0; jdim < NSD; jdim++)
        {
          rateofstrain += two_epsilon(idim, jdim) * two_epsilon(jdim, idim);
        }
      }

      rateofstrain = (sqrt(rateofstrain / 2.0));

      //                                      +-                                 -+ 1
      //                                  2   |          /    \          /   \    | -
      //    visc          = dens * (C_S*h)  * | 2 * eps | vel |   * eps | vel |   | 2
      //        turbulent                     |          \   / ij        \   / ij |
      //                                      +-                                 -+
      //                                      |                                   |
      //                                      +-----------------------------------+
      //                                                   rate of strain
      sgvisc = dens * Cs * Cs * hk_sgvisc * hk_sgvisc * rateofstrain;
    }

    // set parameter in sublist turbulence
    Teuchos::ParameterList* modelparams = &(params.sublist("TURBULENCE MODEL"));
    Teuchos::RCP<std::vector<double>> sum_N_stream =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("local_N_stream_sum");
    Teuchos::RCP<std::vector<double>> sum_N_normal =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("local_N_normal_sum");
    Teuchos::RCP<std::vector<double>> sum_N_span =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("local_N_span_sum");
    Teuchos::RCP<std::vector<double>> sum_B_stream =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("local_B_stream_sum");
    Teuchos::RCP<std::vector<double>> sum_B_normal =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("local_B_normal_sum");
    Teuchos::RCP<std::vector<double>> sum_B_span =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("local_B_span_sum");
    Teuchos::RCP<std::vector<double>> sum_Csgs =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("local_Csgs_sum");
    Teuchos::RCP<std::vector<double>> sum_Nphi;
    Teuchos::RCP<std::vector<double>> sum_Dphi;
    Teuchos::RCP<std::vector<double>> sum_Csgs_phi;
    if (withscatra)
    {
      sum_Nphi = modelparams->get<Teuchos::RCP<std::vector<double>>>("local_Nphi_sum");
      sum_Dphi = modelparams->get<Teuchos::RCP<std::vector<double>>>("local_Dphi_sum");
      sum_Csgs_phi = modelparams->get<Teuchos::RCP<std::vector<double>>>("local_Csgs_phi_sum");
    }
    Teuchos::RCP<std::vector<double>> sum_sgvisc =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("local_sgvisc_sum");

    // the coordinates of the element layers in the channel
    // planecoords are named nodeplanes in turbulence_statistics_channel!
    Teuchos::RCP<std::vector<double>> planecoords =
        modelparams->get<Teuchos::RCP<std::vector<double>>>("planecoords", Teuchos::null);
    if (planecoords == Teuchos::null)
      dserror("planecoords is null, but need channel_flow_of_height_2\n");

    bool found = false;
    int nlayer = 0;
    for (nlayer = 0; nlayer < (int)(*planecoords).size() - 1;)
    {
      if (center < (*planecoords)[nlayer + 1])
      {
        found = true;
        break;
      }
      nlayer++;
    }
    if (found == false)
    {
      dserror("could not determine element layer");
    }

    (*sum_N_stream)[nlayer] += Nvel[0];
    (*sum_N_normal)[nlayer] += Nvel[1];
    (*sum_N_span)[nlayer] += Nvel[2];
    (*sum_B_stream)[nlayer] += B(0, 0);
    (*sum_B_normal)[nlayer] += B(1, 0);
    (*sum_B_span)[nlayer] += B(2, 0);
    (*sum_Csgs)[nlayer] += Csgs;
    if (withscatra)
    {
      (*sum_Csgs_phi)[nlayer] += Csgs_phi;
      (*sum_Nphi)[nlayer] += Nphi;
      (*sum_Dphi)[nlayer] += D;
    }
    (*sum_sgvisc)[nlayer] += sgvisc;

    return;
  }  // DRT::ELEMENTS::Fluid::f3_get_mf_params


  //----------------------------------------------------------------------
  // calculate mean Cai of multifractal subgrid-scale modeling approach
  //                                                       rasthofer 08/12
  //----------------------------------------------------------------------
  template <int NEN, int NSD, CORE::FE::CellType DISTYPE>
  void f3_get_mf_nwc(DRT::ELEMENTS::Fluid* ele, DRT::ELEMENTS::FluidEleParameterStd* fldpara,
      double& Cai, double& vol, std::vector<double>& vel, std::vector<double>& sca,
      const double& thermpress)
  {
    // allocate a fixed size array for nodal velocities an scalars
    CORE::LINALG::Matrix<NSD, NEN> evel;
    CORE::LINALG::Matrix<1, NEN> esca;

    // split velocity and throw away  pressure, insert into element array
    // insert scalar
    for (int inode = 0; inode < NEN; inode++)
    {
      esca(0, inode) = sca[inode];
      for (int idim = 0; idim < NSD; idim++) evel(idim, inode) = vel[inode * 4 + idim];
    }

    if (fldpara->AdaptCsgsPhi())
    {
      // allocate array for gauss-point velocities and derivatives
      CORE::LINALG::Matrix<NSD, 1> velint;
      CORE::LINALG::Matrix<NSD, NSD> velintderxy;

      // allocate arrays for shapefunctions, derivatives and the transposed jacobian
      CORE::LINALG::Matrix<NEN, 1> funct;
      CORE::LINALG::Matrix<NSD, NSD> xjm;
      CORE::LINALG::Matrix<NSD, NSD> xji;
      CORE::LINALG::Matrix<NSD, NEN> deriv;
      CORE::LINALG::Matrix<NSD, NEN> derxy;

      // get node coordinates of element
      CORE::LINALG::Matrix<NSD, NEN> xyze;
      for (int inode = 0; inode < NEN; inode++)
      {
        for (int idim = 0; idim < NSD; idim++) xyze(idim, inode) = ele->Nodes()[inode]->X()[idim];
      }

      // use one-point Gauss rule
      CORE::FE::IntPointsAndWeights<NSD> intpoints(
          DRT::ELEMENTS::DisTypeToStabGaussRule<DISTYPE>::rule);

      // coordinates of the current integration point
      const double* gpcoord = (intpoints.IP().qxg)[0];
      CORE::LINALG::Matrix<NSD, 1> xsi;
      for (int idim = 0; idim < NSD; idim++) xsi(idim) = gpcoord[idim];

      double wquad = intpoints.IP().qwgt[0];

      // shape functions and their first derivatives
      CORE::FE::shape_function<DISTYPE>(xsi, funct);
      CORE::FE::shape_function_deriv1<DISTYPE>(xsi, deriv);

      // get Jacobian matrix and determinant
      xjm.MultiplyNT(deriv, xyze);
      double det = xji.Invert(xjm);
      // check for degenerated elements
      if (det < 1E-16)
        dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", ele->Id(), det);

      // set element volume
      vol = wquad * det;

      // adopt integration points and weights for gauss point evaluation of B
      if (fldpara->BGp())
      {
        CORE::FE::IntPointsAndWeights<NSD> gauss_intpoints(
            DRT::ELEMENTS::DisTypeToOptGaussRule<DISTYPE>::rule);
        intpoints = gauss_intpoints;
      }

      for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
      {
        // coordinates of the current integration point
        const double* gpcoord_iquad = (intpoints.IP().qxg)[iquad];
        for (int idim = 0; idim < NSD; idim++) xsi(idim) = gpcoord_iquad[idim];

        wquad = intpoints.IP().qwgt[iquad];

        // shape functions and their first derivatives
        CORE::FE::shape_function<DISTYPE>(xsi, funct);
        CORE::FE::shape_function_deriv1<DISTYPE>(xsi, deriv);

        // get Jacobian matrix and determinant
        xjm.MultiplyNT(deriv, xyze);
        det = xji.Invert(xjm);
        // check for degenerated elements
        if (det < 1E-16)
          dserror(
              "GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", ele->Id(), det);

        double fac = wquad * det;

        // compute global first derivatives
        derxy.Multiply(xji, deriv);

        // get velocity at integration point
        velint.Multiply(evel, funct);

        // get material
        double dens = 0.0;
        double visc = 0.0;
        Teuchos::RCP<MAT::Material> material = ele->Material();
        if (material->MaterialType() == INPAR::MAT::m_sutherland)
        {
          const MAT::Sutherland* actmat = static_cast<const MAT::Sutherland*>(material.get());

          // compute temperature at gauss point
          double temp = 0.0;
          for (int rr = 0; rr < NEN; rr++) temp += funct(rr, 0) * esca(0, rr);

          // compute density and viscosity based on temperature
          dens = actmat->ComputeDensity(temp, thermpress);
          visc = actmat->ComputeViscosity(temp);
        }
        else if (material->MaterialType() == INPAR::MAT::m_fluid)
        {
          const MAT::NewtonianFluid* actmat =
              static_cast<const MAT::NewtonianFluid*>(material.get());

          // get density and viscosity
          dens = actmat->Density();
          visc = actmat->Viscosity();
        }
        else
          dserror("Newtonian fluid or Sutherland material expected!");

        // calculate characteristic element length
        double hk = 1.0e+10;
        switch (fldpara->RefLength())
        {
          case INPAR::FLUID::streamlength:
          {
            // a) streamlength due to Tezduyar et al. (1992)
            // get norm of velocity
            const double vel_norm = velint.Norm2();
            // normed velocity vector
            CORE::LINALG::Matrix<NSD, 1> velino(true);
            if (vel_norm >= 1e-6)
              velino.Update(1.0 / vel_norm, velint);
            else
            {
              velino.Clear();
              velino(0, 0) = 1.0;
            }
            CORE::LINALG::Matrix<NEN, 1> tmp;
            tmp.MultiplyTN(derxy, velino);
            const double val = tmp.Norm1();
            hk = 2.0 / val;

            break;
          }
          case INPAR::FLUID::sphere_diameter:
          {
            // b) volume-equivalent diameter
            hk = std::pow((6. * vol / M_PI), (1.0 / 3.0)) / sqrt(3.0);

            break;
          }
          case INPAR::FLUID::cube_edge:
          {
            // c) cubic element length
            hk = std::pow(vol, (1.0 / (double(NSD))));
            break;
          }
          case INPAR::FLUID::metric_tensor:
          {
            /*          +-           -+   +-           -+   +-           -+
                        |             |   |             |   |             |
                        |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
                  G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
                   ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                        |    i     j  |   |    i     j  |   |    i     j  |
                        +-           -+   +-           -+   +-           -+
            */
            CORE::LINALG::Matrix<3, 3> G;

            for (int nn = 0; nn < 3; ++nn)
            {
              for (int rr = 0; rr < 3; ++rr)
              {
                G(nn, rr) = xji(nn, 0) * xji(rr, 0);
                for (int mm = 1; mm < 3; ++mm)
                {
                  G(nn, rr) += xji(nn, mm) * xji(rr, mm);
                }
              }
            }

            /*          +----
                         \
                G : G =   +   G   * G
                -   -    /     ij    ij
                -   -   +----
                         i,j
            */
            double normG = 0;
            for (int nn = 0; nn < 3; ++nn)
            {
              for (int rr = 0; rr < 3; ++rr)
              {
                normG += G(nn, rr) * G(nn, rr);
              }
            }
            hk = std::pow(normG, -0.25);

            break;
          }
          case INPAR::FLUID::gradient_based:
          {
            velintderxy.MultiplyNT(evel, derxy);
            CORE::LINALG::Matrix<3, 1> normed_velgrad;

            for (int rr = 0; rr < 3; ++rr)
            {
              normed_velgrad(rr) = sqrt(velintderxy(0, rr) * velintderxy(0, rr) +
                                        velintderxy(1, rr) * velintderxy(1, rr) +
                                        velintderxy(2, rr) * velintderxy(2, rr));
            }
            double norm = normed_velgrad.Norm2();

            // normed gradient
            if (norm > 1e-6)
            {
              for (int rr = 0; rr < 3; ++rr) normed_velgrad(rr) /= norm;
            }
            else
            {
              normed_velgrad(0) = 1.;
              for (int rr = 1; rr < 3; ++rr) normed_velgrad(rr) = 0.0;
            }

            // get length in this direction
            double val = 0.0;
            for (int rr = 0; rr < NEN; ++rr)
            {
              val += fabs(normed_velgrad(0) * derxy(0, rr) + normed_velgrad(1) * derxy(1, rr) +
                          normed_velgrad(2) * derxy(2, rr));
            }

            hk = 2.0 / val;

            break;
          }
          default:
            dserror("Unknown length");
        }

        // calculate norm of strain rate
        double strainnorm = 0.0;
        // compute (resolved) norm of strain
        //
        //          +-                                 -+ 1
        //          |          /   \           /   \    | -
        //          |     eps | vel |   * eps | vel |   | 2
        //          |          \   / ij        \   / ij |
        //          +-                                 -+
        //
        velintderxy.MultiplyNT(evel, derxy);
        CORE::LINALG::Matrix<NSD, NSD> twoeps;
        for (int idim = 0; idim < NSD; idim++)
        {
          for (int jdim = 0; jdim < NSD; jdim++)
            twoeps(idim, jdim) = velintderxy(idim, jdim) + velintderxy(jdim, idim);
        }
        for (int idim = 0; idim < NSD; idim++)
        {
          for (int jdim = 0; jdim < NSD; jdim++)
            strainnorm += twoeps(idim, jdim) * twoeps(idim, jdim);
        }
        strainnorm = (sqrt(strainnorm / 4.0));

        // get Re from strain rate
        double Re_ele_str = strainnorm * hk * hk * dens / visc;
        if (Re_ele_str < 0.0) dserror("Something went wrong!");
        // ensure positive values
        if (Re_ele_str < 1.0) Re_ele_str = 1.0;

        // calculate corrected Cai
        //           -3/16
        //   (1 - (Re)   )
        //
        Cai += (1.0 - pow(Re_ele_str, -3.0 / 16.0)) * fac;
      }
    }

    return;
  }


  //-----------------------------------------------------------------------
  // free-suface flows
  //-----------------------------------------------------------------------


  /*!
    \brief Calculate node normals using the volume integral in Wall (7.13).
    That formula considers all surfaces of the element, not only the
    free surfaces.

    \param elevec1 (out)      : grad(N) integrated over Element
    \param edispnp (in)       : Displacement-vector
  */
  template <CORE::FE::CellType DISTYPE>
  void ElementNodeNormal(DRT::ELEMENTS::Fluid* ele, Teuchos::ParameterList& params,
      DRT::Discretization& discretization, std::vector<int>& lm,
      CORE::LINALG::SerialDenseVector& elevec1)
  {
    // this evaluates the node normals using the volume integral in Wall
    // (7.13). That formula considers all surfaces of the element, not only the
    // free surfaces. This causes difficulties because the free surface normals
    // point outwards on nodes at the rim of a basin (e.g. channel-flow).

    // get number of nodes
    const int iel = CORE::FE::num_nodes<DISTYPE>;

    // get number of dimensions
    const int nsd = CORE::FE::dim<DISTYPE>;

    // get number of dof's
    const int numdofpernode = nsd + 1;


    /*
    // create matrix objects for nodal values
    CORE::LINALG::Matrix<3,iel>       edispnp;

    if (is_ale_)
    {
      // get most recent displacements
      Teuchos::RCP<const Epetra_Vector> dispnp
        =
        discretization.GetState("dispnp");

      if (dispnp==Teuchos::null)
      {
        dserror("Cannot get state vector 'dispnp'");
      }

      std::vector<double> mydispnp(lm.size());
      CORE::FE::ExtractMyValues(*dispnp,mydispnp,lm);

      // extract velocity part from "mygridvelaf" and get
      // set element displacements
      for (int i=0;i<iel;++i)
      {
        int fi    =4*i;
        int fip   =fi+1;
        int fipp  =fip+1;
        edispnp(0,i)    = mydispnp   [fi  ];
        edispnp(1,i)    = mydispnp   [fip ];
        edispnp(2,i)    = mydispnp   [fipp];
      }
    }

    // set element data
    const CORE::FE::CellType distype = this->Shape();
  */
    //----------------------------------------------------------------------------
    //                         ELEMENT GEOMETRY
    //----------------------------------------------------------------------------
    // CORE::LINALG::Matrix<3,iel>  xyze;
    CORE::LINALG::Matrix<nsd, iel> xyze;

    // get node coordinates
    // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
    // element!)
    CORE::GEO::fillInitialPositionArray<DISTYPE, nsd, CORE::LINALG::Matrix<nsd, iel>>(ele, xyze);

    /*
    // get node coordinates
    DRT::Node** nodes = Nodes();
    for (int inode=0; inode<iel; inode++)
    {
      const auto& x = nodes[inode]->X();
      xyze(0,inode) = x[0];
      xyze(1,inode) = x[1];
      xyze(2,inode) = x[2];
    }
  */
    if (ele->IsAle())
    {
      // --------------------------------------------------
      // create matrix objects for nodal values
      CORE::LINALG::Matrix<nsd, iel> edispnp(true);

      // get most recent displacements
      Teuchos::RCP<const Epetra_Vector> dispnp = discretization.GetState("dispnp");

      if (dispnp == Teuchos::null)
      {
        dserror("Cannot get state vector 'dispnp'");
      }

      std::vector<double> mydispnp(lm.size());
      CORE::FE::ExtractMyValues(*dispnp, mydispnp, lm);

      // extract velocity part from "mygridvelaf" and get
      // set element displacements
      for (int inode = 0; inode < iel; ++inode)
      {
        for (int idim = 0; idim < nsd; ++idim)
        {
          edispnp(idim, inode) = mydispnp[numdofpernode * inode + idim];
        }
      }
      // get new node positions for isale
      xyze += edispnp;
    }

    /*
    // add displacement, when fluid nodes move in the ALE case
    if (is_ale_)
    {
      for (int inode=0; inode<iel; inode++)
      {
        xyze(0,inode) += edispnp(0,inode);
        xyze(1,inode) += edispnp(1,inode);
        xyze(2,inode) += edispnp(2,inode);
      }
    }
  */
    //------------------------------------------------------------------
    //                       INTEGRATION LOOP
    //------------------------------------------------------------------
    // CORE::LINALG::Matrix<iel,1  > funct;
    // CORE::LINALG::Matrix<3,  iel> deriv;
    // CORE::LINALG::Matrix<3,  3  > xjm;
    // CORE::LINALG::Matrix<3,  3  > xji;

    CORE::LINALG::Matrix<iel, 1> funct(true);
    CORE::LINALG::Matrix<nsd, iel> deriv(true);
    // CORE::LINALG::Matrix<6,6>   bm(true);

    // get Gaussrule
    const CORE::FE::IntPointsAndWeights<nsd> intpoints(
        DRT::ELEMENTS::DisTypeToOptGaussRule<DISTYPE>::rule);

    // gaussian points
    // const GaussRule3D          gaussrule = getOptimalGaussrule(distype);
    // const IntegrationPoints3D  intpoints(gaussrule);

    for (int iquad = 0; iquad < intpoints.IP().nquad; ++iquad)
    {
      // local Gauss point coordinates
      CORE::LINALG::Matrix<nsd, 1> xsi(true);

      // local coordinates of the current integration point
      const double* gpcoord = (intpoints.IP().qxg)[iquad];
      for (int idim = 0; idim < nsd; ++idim)
      {
        xsi(idim) = gpcoord[idim];
      }
      /*
          // set gauss point coordinates
          CORE::LINALG::Matrix<3,1> gp;

          gp(0)=intpoints.qxg[iquad][0];
          gp(1)=intpoints.qxg[iquad][1];
          gp(2)=intpoints.qxg[iquad][2];

          if(!(distype == CORE::FE::CellType::nurbs8
               ||
               distype == CORE::FE::CellType::nurbs27))
          {
            // get values of shape functions and derivatives in the gausspoint
            CORE::FE::shape_function_3D       (funct,gp(0),gp(1),gp(2),distype);
            CORE::FE::shape_function_3D_deriv1(deriv,gp(0),gp(1),gp(2),distype);
          }
          else
          {
            dserror("not implemented");
          }
      */

      if (not DRT::ELEMENTS::IsNurbs<DISTYPE>::isnurbs)
      {
        // get values of shape functions and derivatives in the gausspoint
        // CORE::FE::shape_function_3D       (funct,gp(0),gp(1),gp(2),distype);
        // CORE::FE::shape_function_3D_deriv1(deriv,gp(0),gp(1),gp(2),distype);

        // shape function derivs of boundary element at gausspoint
        CORE::FE::shape_function<DISTYPE>(xsi, funct);
        CORE::FE::shape_function_deriv1<DISTYPE>(xsi, deriv);
      }
      else
        dserror("Nurbs are not implemented yet");

      CORE::LINALG::Matrix<nsd, nsd> xjm(true);
      CORE::LINALG::Matrix<nsd, nsd> xji(true);

      // compute jacobian matrix
      // determine jacobian at point r,s,t
      xjm.MultiplyNT(deriv, xyze);

      // determinant and inverse of jacobian
      const double det = xji.Invert(xjm);

      // check for degenerated elements
      if (det < 0.0)
      {
        dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", ele->Id(), det);
      }

      // set total integration factor
      const double fac = intpoints.IP().qwgt[iquad] * det;

      // integrate shapefunction gradient over element
      for (int dim = 0; dim < 3; dim++)
      {
        for (int node = 0; node < iel; node++)
        {
          elevec1[4 * node + dim] += (deriv(0, node) * xji(dim, 0) + deriv(1, node) * xji(dim, 1) +
                                         deriv(2, node) * xji(dim, 2)) *
                                     fac;
        }
      }
    }

    return;
  }


}  // namespace FLD

BACI_NAMESPACE_CLOSE

#endif