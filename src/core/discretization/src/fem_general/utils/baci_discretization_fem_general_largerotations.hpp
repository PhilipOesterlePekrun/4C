/*----------------------------------------------------------------------------*/
/*! \file

\brief A set of utility functions for large rotations

\level 2

*/
/*----------------------------------------------------------------------------*/

#ifndef FOUR_C_DISCRETIZATION_FEM_GENERAL_LARGEROTATIONS_HPP
#define FOUR_C_DISCRETIZATION_FEM_GENERAL_LARGEROTATIONS_HPP

#include "baci_config.hpp"

#include "baci_linalg_fixedsizematrix.hpp"
#include "baci_utils_fad.hpp"

BACI_NAMESPACE_OPEN


namespace CORE::LARGEROTATIONS
{
  /**
   * \brief Compute spin matrix S from rotation angle vector theta, Crisfield, Vol. 2, equation
   * (16.8)
   * @param S (out) spin matrix
   * @param theta (in) rotation angle vector
   */
  template <typename T>
  void computespin(CORE::LINALG::Matrix<3, 3, T>& S, const CORE::LINALG::Matrix<3, 1, T>& theta)
  {
    // function based on Crisfield Vol. 2, Section 16 (16.8)
    S(0, 0) = 0.0;
    S(0, 1) = -theta(2);
    S(0, 2) = theta(1);
    S(1, 0) = theta(2);
    S(1, 1) = 0.0;
    S(1, 2) = -theta(0);
    S(2, 0) = -theta(1);
    S(2, 1) = theta(0);
    S(2, 2) = 0.0;
  }

  /**
   * \brief Compute rotation matrix R from quaternion q, Crisfield, Vol. 2, equation (16.70)
   * @param q (in) quaternion
   * @param R (out) rotation matrix
   */
  template <typename T>
  void quaterniontotriad(const CORE::LINALG::Matrix<4, 1, T>& q, CORE::LINALG::Matrix<3, 3, T>& R)
  {
    // separate storage of vector part of q
    CORE::LINALG::Matrix<3, 1, T> qvec;
    for (int i = 0; i < 3; i++) qvec(i) = q(i);

    // setting R to third summand of equation (16.70)
    CORE::LARGEROTATIONS::computespin(R, qvec);
    R.Scale(2 * q(3));

    // adding second summand of equation (16.70)
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) R(i, j) += 2 * q(i) * q(j);

    // adding diagonal entries according to first summand of equation (16.70)
    R(0, 0) = 1 - 2 * (q(1) * q(1) + q(2) * q(2));
    R(1, 1) = 1 - 2 * (q(0) * q(0) + q(2) * q(2));
    R(2, 2) = 1 - 2 * (q(0) * q(0) + q(1) * q(1));

    return;
  }

  /**
   * \brief Compute quaternion q from rotation angle theta, Crisfield, Vol. 2, equation (16.67)
   * @param theta (in) rotation angle
   * @param q (out) quaternion
   */
  template <typename T>
  void angletoquaternion(
      const CORE::LINALG::Matrix<3, 1, T>& theta, CORE::LINALG::Matrix<4, 1, T>& q)
  {
    // absolute value of rotation angle theta
    const T abs_theta = CORE::FADUTILS::VectorNorm(theta);

    // computing quaterion for rotation by angle theta, Crisfield, Vol. 2, equation (16.67)
    if (abs_theta > 1e-12)
    {
      q(0) = theta(0) * sin(abs_theta / 2) / abs_theta;
      q(1) = theta(1) * sin(abs_theta / 2) / abs_theta;
      q(2) = theta(2) * sin(abs_theta / 2) / abs_theta;
      q(3) = cos(abs_theta / 2);
    }
    else
    {
      // This is compiled for FAD types
      // altered due to FAD - with theta->0 the equations above simplify in the manner below
      // NOTE: the equations below are chosen in a way that the linearisation (with FAD) is the
      // same of the equations above for the case theta->0
      q(0) = 0.5 * theta(0);
      q(1) = 0.5 * theta(1);
      q(2) = 0.5 * theta(2);
      q(3) = cos(abs_theta / 2);
    }

    return;
  }

  /**
   * \brief Compute triad from angle theta, Crisfield Vol. 2, equation (16.22)
   * @param theta (in) rotation angle
   * @param triad (out) triad
   */
  template <typename T>
  void angletotriad(
      const CORE::LINALG::Matrix<3, 1, T>& theta, CORE::LINALG::Matrix<3, 3, T>& triad)
  {
    triad.Clear();
    CORE::LINALG::Matrix<4, 1, T> quaternion(true);
    CORE::LARGEROTATIONS::angletoquaternion(theta, quaternion);
    CORE::LARGEROTATIONS::quaterniontotriad(quaternion, triad);
  }

  /**
   * \brief Compute rotation angle vector theta from quaternion q, Crisfield, Vol. 2, equation
   * (16.67)
   *
   * The following function computes from a quaternion q an angle theta within [-PI; PI]; such an
   * interval is imperative for the use of the resulting angle together with formulae like
   * Crisfield, Vol. 2, equation (16.90); note that these formulae comprise not only trigonometric
   * functions, but rather the angle theta directly. Hence they are not 2*PI-invariant!!!
   *
   * @param q (in) quaternion
   * @param theta (out) rotatin angle
   */
  template <typename T>
  void quaterniontoangle(
      const CORE::LINALG::Matrix<4, 1, T>& q, CORE::LINALG::Matrix<3, 1, T>& theta)
  {
    // if the rotation angle is pi we have q(3) == 0 and the rotation angle vector can be computed
    // by
    if (q(3) == 0)
    {
      // note that with q(3) == 0 the first three elements of q represent the unit direction vector
      // of the angle according to Crisfield, Vol. 2, equation (16.67)
      for (int i = 0; i < 3; i++) theta(i) = q(i) * M_PI;
    }
    else
    {
      // otherwise the angle can be computed from a quaternion via Crisfield, Vol. 2, eq. (16.79)
      CORE::LINALG::Matrix<3, 1, T> omega;
      for (int i = 0; i < 3; i++)
      {
        omega(i) = q(i) * 2 / q(3);
      }
      const T abs_omega = CORE::FADUTILS::VectorNorm(omega);
      const T tanhalf = abs_omega / 2;
      const T thetaabs = atan(tanhalf) * 2;

      // if the rotation angle is zero we return a zero rotation angle vector at once
      if (abs_omega < 1e-12)
      {
        // altered due to FAD, the equations simplify in that way for theta->0
        for (int i = 0; i < 3; i++)
          theta(i) =
              2.0 * q(i);  // altered due to FAD, the equations simplify in that way for theta->0
      }
      else
      {
        theta = omega;
        theta.Scale(thetaabs / abs_omega);
      }
    }

    return;
  }

  /**
   * \brief Compute from quaternion q the Rodrigues parameters omega, Crisfield, Vol. 2, equation
   * (16.79)
   * @param q (in) quaternion)
   * @param omega (out) rodrigures angles
   */
  template <typename T>
  void quaterniontorodrigues(
      const CORE::LINALG::Matrix<4, 1, T>& q, CORE::LINALG::Matrix<3, 1, T>& omega)
  {
    /* the Rodrigues parameters are defined only for angles whose absolute valued is smaller than
     * PI, i.e. for which the fourth component of the quaternion is unequal zero; if this is not
     * satisfied for the quaternion passed into this method an error is thrown*/
    if (q(3) == 0)
      dserror("cannot compute Rodrigues parameters for angles with absolute valued PI !!!");

    // in any case except for the one dealt with above the angle can be computed from a
    // quaternion via Crisfield, Vol. 2, eq. (16.79)
    for (int i = 0; i < 3; i++) omega(i) = q(i) * 2 / q(3);

    return;
  }


  /**
   * \brief Compute quaternion q from a rotation matrix R, Crisfield, Vol. 2, equation (16.74) -
   * (16.78)
   * @param R (in) rotation matrix
   * @param q (out) quaternion
   */
  template <typename T>
  void triadtoquaternion(const CORE::LINALG::Matrix<3, 3, T>& R, CORE::LINALG::Matrix<4, 1, T>& q)
  {
    T trace = R(0, 0) + R(1, 1) + R(2, 2);

    if (trace > R(0, 0) && trace > R(1, 1) && trace > R(2, 2))
    {
      q(3) = 0.5 * CORE::FADUTILS::sqrt<T>(1 + trace);
      /*note: if trace is greater than each element on diagonal, all diagonal elements are
       *positive and hence also the trace is positive and thus q(3) > 0 so that division by q(3)
       *is allowed*/
      q(0) = (R(2, 1) - R(1, 2)) / (4 * q(3));
      q(1) = (R(0, 2) - R(2, 0)) / (4 * q(3));
      q(2) = (R(1, 0) - R(0, 1)) / (4 * q(3));
    }
    else
    {
      for (int i = 0; i < 3; i++)
      {
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;

        if (R(i, i) >= R(j, j) && R(i, i) >= R(k, k))
        {
          // equation (16.78a)
          q(i) = CORE::FADUTILS::sqrt<T>(0.5 * R(i, i) + 0.25 * (1 - trace));

          // equation (16.78b)
          q(3) = 0.25 * (R(k, j) - R(j, k)) / q(i);

          // equation (16.78c)
          q(j) = 0.25 * (R(j, i) + R(i, j)) / q(i);
          q(k) = 0.25 * (R(k, i) + R(i, k)) / q(i);
        }
      }
    }

    return;
  }

  /**
   * \brief Quaternion product q12 = q2*q1, Crisfield, Vol. 2, equation (16.71)
   *
   * If q1 and q2 correspond to the rotation matrices R1 and R2, respectively, the compound
   * rotation R12 = R2*R1 corresponds to the compound quaternion q12 = q2*q1.
   *
   * @tparam T1, T2, T3 Template parameter of the individual quaternions.
   * @param q1 (in) The first quaternion to apply.
   * @param q2 (in) The second quaternion to apply.
   * @param q12 (out) The compound quaternion.
   */
  template <class T1, class T2, class T3>
  void quaternionproduct(const T1& q1, const T2& q2, T3& q12)
  {
    dsassert(q12.M() == 4 and q12.N() == 1 and q1.M() == 4 and q1.N() == 1 and q2.M() == 4 and
                 q2.N() == 1,
        "size mismatch: expected 4x1 vector for quaternion");

    q12(0) = q2(3) * q1(0) + q1(3) * q2(0) + q2(1) * q1(2) - q1(1) * q2(2);
    q12(1) = q2(3) * q1(1) + q1(3) * q2(1) + q2(2) * q1(0) - q1(2) * q2(0);
    q12(2) = q2(3) * q1(2) + q1(3) * q2(2) + q2(0) * q1(1) - q1(0) * q2(1);
    q12(3) = q2(3) * q1(3) - q2(2) * q1(2) - q2(1) * q1(1) - q2(0) * q1(0);
  }

  /**
   * \brief Compute inverse quaternion q^{-1} of quaternion q
   * @param q (in) quaternion
   * @return inversed quaternion
   */
  template <typename T>
  CORE::LINALG::Matrix<4, 1, T> inversequaternion(const CORE::LINALG::Matrix<4, 1, T>& q)
  {
    // square norm ||q||^2 of quaternion q
    const T qnormsq = CORE::FADUTILS::VectorNorm(q);

    // declaration of variable for inverse quaternion
    CORE::LINALG::Matrix<4, 1, T> qinv;

    // inverse quaternion q^(-1) = [-q0, -q1, -q2, q3] / ||q||^2;
    for (int i = 0; i < 3; i++) qinv(i) = -q(i) / qnormsq;

    qinv(3) = q(3) / qnormsq;

    return qinv;
  }

  /**
   * \brief Compute matrix T(\theta) from Jelenic 1999, eq. (2.5), equivalent to matrix H^{-1} in
   * Crisfield, (16.94)
   * @param theta (in) rotation angle
   * @return transformation matrix
   */
  template <typename T>
  CORE::LINALG::Matrix<3, 3, T> Tmatrix(CORE::LINALG::Matrix<3, 1, T> theta)
  {
    CORE::LINALG::Matrix<3, 3, T> result(true);
    const T theta_abs = CORE::FADUTILS::VectorNorm(theta);

    // in case of theta_abs == 0 the following computation has problems with singularities
    if (theta_abs > 1e-8)
    {
      computespin(result, theta);
      result.Scale(-0.5);

      T theta_abs_half = theta_abs / 2.0;

      for (int i = 0; i < 3; i++) result(i, i) += theta_abs / (2.0 * tan(theta_abs_half));

      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          result(i, j) += theta(i) * theta(j) * (1.0 - theta_abs / (2.0 * tan(theta_abs_half))) /
                          (theta_abs * theta_abs);
    }
    // based on the small angle approximation tan(x)=x, we get: T = I - 0.5*S(theta)
    else
    {
      CORE::LARGEROTATIONS::computespin(result, theta);
      result.Scale(-0.5);
      for (int j = 0; j < 3; j++) result(j, j) += 1.0;
    }
    return result;
  }


  /**
   * \brief Compute  matrix T(\theta)^{-1} from Jelenic 1999, eq. (2.5)
   * @param theta (in) rotation angle
   * @return inverse of transformation matrix
   */
  template <typename T>
  CORE::LINALG::Matrix<3, 3, T> Tinvmatrix(CORE::LINALG::Matrix<3, 1, T> theta)
  {
    CORE::LINALG::Matrix<3, 3, T> result(true);
    T theta_abs =
        CORE::FADUTILS::sqrt<T>(theta(0) * theta(0) + theta(1) * theta(1) + theta(2) * theta(2));

    // in case of theta_abs == 0 the following computation has problems with ill-conditioning /
    // singularities
    if (theta_abs > 1e-8)
    {
      // ultimate term in eq. (2.5)
      CORE::LARGEROTATIONS::computespin(result, theta);
      result.Scale((1.0 - cos(theta_abs)) / (theta_abs * theta_abs));

      // penultimate term in eq. (2.5)
      for (int i = 0; i < 3; i++) result(i, i) += sin(theta_abs) / (theta_abs);

      // first term on the right side in eq. (2.5)
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          result(i, j) +=
              theta(i) * theta(j) * (1.0 - sin(theta_abs) / (theta_abs)) / (theta_abs * theta_abs);
    }
    // in case of theta_abs == 0 H(theta) is the identity matrix and hence also Hinv
    else
    {
      // based on the small angle approximation sin(x)=x and 1-cos(x)=x^2/2 we get: Tinv = I +
      // 0.5*S(theta) -> Tinv'=0.5*S(theta')
      CORE::LARGEROTATIONS::computespin(result, theta);
      result.Scale(0.5);
      for (int j = 0; j < 3; j++) result(j, j) += 1.0;
    }

    return result;
  }

  /**
   * \brief Compute compute dT^{-1}(\theta)/dx according to the two-lined equation below (3.19) on
   * page 152 of Jelenic 1999
   */
  template <typename T>
  void computedTinvdx(const CORE::LINALG::Matrix<3, 1, T>& Psil,
      const CORE::LINALG::Matrix<3, 1, T>& Psilprime, CORE::LINALG::Matrix<3, 3, T>& dTinvdx)
  {
    // auxiliary matrix for storing intermediate results
    CORE::LINALG::Matrix<3, 3, T> auxmatrix;

    // norm of \Psi^l:
    T normPsil = CORE::FADUTILS::sqrt<T>(Psil(0) * Psil(0) + Psil(1) * Psil(1) + Psil(2) * Psil(2));

    // for relative rotations smaller then 1e-12 we use the limit for Psil -> 0 according to the
    // comment above NOTE 4 on page 152, Jelenic 1999
    if (normPsil < 1e-8)
    {
      computespin(dTinvdx, Psilprime);
      dTinvdx.Scale(0.5);
    }
    else
    {
      // scalarproduct \Psi^{l,t} \cdot \Psi^{l,'}
      T scalarproductPsilPsilprime = 0.0;
      for (int i = 0; i < 3; i++) scalarproductPsilPsilprime += Psil(i) * Psilprime(i);

      // spin matrices of Psil and Psilprime
      CORE::LINALG::Matrix<3, 3, T> spinPsil;
      CORE::LINALG::Matrix<3, 3, T> spinPsilprime;
      computespin(spinPsil, Psil);
      computespin(spinPsilprime, Psilprime);

      // third summand
      dTinvdx.Multiply(spinPsilprime, spinPsil);
      auxmatrix.Multiply(spinPsil, spinPsilprime);
      dTinvdx += auxmatrix;
      dTinvdx.Scale((1.0 - sin(normPsil) / normPsil) / (normPsil * normPsil));

      // first summand
      auxmatrix.PutScalar(0.0);
      auxmatrix += spinPsil;
      auxmatrix.Scale(scalarproductPsilPsilprime *
                      (normPsil * sin(normPsil) - 2 * (1.0 - cos(normPsil))) /
                      (normPsil * normPsil * normPsil * normPsil));
      dTinvdx += auxmatrix;

      // second summand
      auxmatrix.PutScalar(0.0);
      auxmatrix += spinPsilprime;
      auxmatrix.Scale((1.0 - cos(normPsil)) / (normPsil * normPsil));
      dTinvdx += auxmatrix;

      // fourth summand
      auxmatrix.Multiply(spinPsil, spinPsil);
      auxmatrix.Scale(scalarproductPsilPsilprime *
                      (3.0 * sin(normPsil) - normPsil * (2.0 + cos(normPsil))) /
                      (normPsil * normPsil * normPsil * normPsil * normPsil));
      dTinvdx += auxmatrix;
    }

    return;
  }

  //! Transformation from node number according to Crisfield 1999 to storage position applied in
  //! BACI
  unsigned int NumberingTrafo(const unsigned int j, const unsigned int numnode);

  //! Rotate an arbitrary triad around its first base vector (tangent)
  template <typename T>
  void RotateTriad(const CORE::LINALG::Matrix<3, 3, T>& triad, const T& alpha,
      CORE::LINALG::Matrix<3, 3, T>& triad_rot)
  {
    for (int i = 0; i < 3; i++)
    {
      triad_rot(i, 0) = triad(i, 0);
      triad_rot(i, 1) = triad(i, 1) * cos(alpha) + triad(i, 2) * sin(alpha);
      triad_rot(i, 2) = triad(i, 2) * cos(alpha) - triad(i, 1) * sin(alpha);
    }

    return;
  }

  //! Calculate the SR mapping for a given reference system triad_ref and a given tangent vector
  //! r_s
  template <typename T>
  void CalculateSRTriads(const CORE::LINALG::Matrix<3, 1, T>& r_s,
      const CORE::LINALG::Matrix<3, 3, T>& triad_ref, CORE::LINALG::Matrix<3, 3, T>& triad)
  {
    // In this calculation, r_s does not necessarily have to be a unit vector.
    T temp_scalar1 = 0.0;
    T temp_scalar2 = 0.0;
    T temp_scalar3 = 0.0;
    T fac_n0 = 0.0;
    T fac_b0 = 0.0;
    T abs_r_s = CORE::FADUTILS::Norm<T>(r_s);

    for (int i = 0; i < 3; i++)
    {
      temp_scalar1 += triad_ref(i, 1) * r_s(i);
      temp_scalar2 += triad_ref(i, 2) * r_s(i);
      temp_scalar3 += triad_ref(i, 0) * r_s(i);
    }

    // This line is necessary in order to avoid a floating point exception and subsequent
    // abortion of the simulation in case the denominator equals zero. In such a case,
    // convergence is still unlikely. However, since the simulation can continue,
    // functionalities such as DIVERCONT==adapt_step will still work!
    if (std::fabs(abs_r_s + temp_scalar3) < 1.0e-10)
    {
      // Todo @grill: think about using runtime error here
      std::cout << "Warning: SR-Denominator scaled!" << std::endl;
      temp_scalar3 = 0.99 * temp_scalar3;
    }

    fac_n0 = temp_scalar1 / (abs_r_s + temp_scalar3);
    fac_b0 = temp_scalar2 / (abs_r_s + temp_scalar3);

    for (int i = 0; i < 3; i++)
    {
      triad(i, 0) = r_s(i) / abs_r_s;
      triad(i, 1) = triad_ref(i, 1) - fac_n0 * (r_s(i) / abs_r_s + triad_ref(i, 0));
      triad(i, 2) = triad_ref(i, 2) - fac_b0 * (r_s(i) / abs_r_s + triad_ref(i, 0));
    }

    return;
  }

  //! compute the relative angle theta between triad_ref and triad so that exp(theta) =
  //! triad_ref^T*triad -> inversion of the right translation triad=triad_ref*exp(theta)
  template <typename T>
  void triadtoangleright(CORE::LINALG::Matrix<3, 1, T>& theta,
      const CORE::LINALG::Matrix<3, 3, T>& triad_ref, const CORE::LINALG::Matrix<3, 3, T>& triad)
  {
    CORE::LINALG::Matrix<3, 3, T> rotation_matrix;
    CORE::LINALG::Matrix<4, 1, T> quaternion;

    rotation_matrix.Clear();
    quaternion.Clear();

    rotation_matrix.MultiplyTN(triad_ref, triad);
    CORE::LARGEROTATIONS::triadtoquaternion(rotation_matrix, quaternion);
    CORE::LARGEROTATIONS::quaterniontoangle(quaternion, theta);
  }

  //! compute the relative angle theta between triad_ref and triad so that exp(theta) =
  //! triad*triad_ref^T -> inversion of the right translation triad=exp(theta)*triad_ref
  template <typename T>
  void triadtoangleleft(CORE::LINALG::Matrix<3, 1, T>& theta,
      const CORE::LINALG::Matrix<3, 3, T>& triad_ref, const CORE::LINALG::Matrix<3, 3, T>& triad)
  {
    CORE::LINALG::Matrix<3, 3, T> rotation_matrix;
    CORE::LINALG::Matrix<4, 1, T> quaternion;

    rotation_matrix.Clear();
    quaternion.Clear();

    rotation_matrix.MultiplyNT(triad, triad_ref);
    CORE::LARGEROTATIONS::triadtoquaternion(rotation_matrix, quaternion);
    CORE::LARGEROTATIONS::quaterniontoangle(quaternion, theta);
  }

}  // namespace CORE::LARGEROTATIONS

BACI_NAMESPACE_CLOSE

#endif