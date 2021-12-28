import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class TECOARM1(DHRobot):
    """
    Class that models a TECO TECOARM1 manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.TECOARM1()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

    .. codeauthor:: Sheng Kai Yang
    """  # noqa

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        # robot length values (metres)
        a = [0, 0, -0.314, -0.284, 0, 0]
        d = [0.1301, 0, 0, 0.1145, 0.090, 0.048]

        alpha = [pi/2, zero, zero, pi/2, -pi/2, zero]

        # mass data, no inertia available
        mass = [0.726332273353425, 0.0933250022663036, 0.579432786411222, 0.667529701346893, 0.512164356061208, 0.47307]
        center_of_mass = [
                [7.75478811526881E-06, -0.000599165473154041, 0.0324781805447384], # Base
                [-0.000387837294488029, 0.000216404599264787, 0.0521144858614554], # A1_Link
                [-0.212847469480224, 0.000498088728510149, 0.124066444887415], # A2_Link
                [-0.306031466342084, 4.70438275923801E-05, -0.0140438274762813], # A3_Link
                [0.000680246030960285, 0.0154565305379361, 0.00330850907185035], # A4_Link
                [-7.1401E-05, -0.017408, 5.9362E-05] # A5_Link
            ]
            # [-1.26029353286761E-07 -0.000181212331748337 -0.00375000000286163] # A6_Link
            # mass 0.0372491118430775
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j],
                a=a[j],
                alpha=alpha[j],
                m=mass[j],
                r=center_of_mass[j],
                G=1
            )
            links.append(link)
    
        super().__init__(
            links,
            name="TECO1",
            manufacturer="TECO",
            keywords=('dynamics', 'symbolic'),
            symbolic=symbolic
        )
    
        # zero angles
        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))
        # horizontal along the x-axis
        self.addconfiguration("qr", np.r_[180, 0, 0, 0, 90, 0]*deg)

        # straight and horizontal
        self.addconfiguration("qs", np.array([0, 0, -pi/2, 0, 0, 0]))

        # nominal table top picking pose
        self.addconfiguration("qn", np.array([0, pi/4, pi, 0, pi/4, 0]))
if __name__ == '__main__':    # pragma nocover

    teco = TECOARM1(symbolic=False)
    print(teco)
    # print(teco.dyntable())
