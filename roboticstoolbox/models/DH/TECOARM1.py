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
        base = 26.45 * inch    # from mounting surface to shoulder axis
        # TODO: add dynamics parameter data
        # robot length values (metres)
        # % theta    d        a        alpha     offset
        # L(1)=Link([0       0.08916    0        pi/2      0     ],'standard'); 
        # L(1).m = 3.7000; %質心
        # L(1).r = [0,-0.02561, 0.00193]; % 鏈接 COG wrt 鏈接坐標系 3x1
        # L(1).I = [0.010267495893, 0.010267495893, 0.00666, 0, 0, 0];% 鏈接慣性矩陣，對稱 3x3，關於鏈接 COG。
        # L(1).G = 100; %齒輪比
        # L(1).Jm = 0.0; % 執行器：電機慣量（電機參考） 

        # L(2)=Link([0       0        0.425     0         0     ],'standard');
        # L(2).m = 8.3930;
        # L(2).r = [0.2125, 0, 0.11336];
        # L(2).I = [0.22689067591, 0.22689067591, 0.0151074, 0, 0, 0];
        # L(2).G = 100;
        # L(2).Jm = 0.0;

        # L(3)=Link([0       0        0.39225   0         0     ],'standard');
        # L(3).m = 2.33;
        # L(3).r = [0.15, 0, 0.0265];
        # L(3).I = [0.049443313556, 0.049443313556, 0.004095, 0, 0, 0];
        # L(3).G = 100;
        # L(3).Jm = 0.0;

        # L(4)=Link([0       0.10915   0        pi/2      0     ],'standard');
        # L(4).m = 1.2190;
        # L(4).r = [0, -0.0018, 0.01634];
        # L(4).I = [0.111172755531, 0.111172755531, 0.21942, 0, 0, 0];
        # L(4).G = 100;
        # L(4).Jm = 0.0;

        # L(5)=Link([0       0.09456    0        -pi/2     0     ],'standard');
        # L(5).m = 1.2190;
        # L(5).r = [0, -0.0018, 0.01634];
        # L(5).I = [0.111172755531, 0.111172755531, 0.21942, 0, 0, 0];
        # L(5).G = 100;
        # L(5).Jm = 0.0;

        # L(6)=Link([0       0.0823     0        0         0     ],'standard');
        # L(6).m = 0.1897;
        # L(6).r = [0, 0, -0.001159];
        # L(6).I = [0.0171364731454, 0.0171364731454, 0.033822, 0, 0, 0];
        # L(6).G = 100; 
        # L(6).Jm = 0.0;


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
        Jm=200e-6,    # actuator inertia
        G=[80, 80, 80, 50, 50, 50]   # gear ratio
        B=[0, 0, 0, 0, 0, 0], # actuator viscous friction coefficient (measured
        # at the motor)
        I=[[0, 0.35, 0, 0, 0, 0],[0, 0.35, 0, 0, 0, 0],[0, 0.35, 0, 0, 0, 0],[0, 0.35, 0, 0, 0, 0],[0, 0.35, 0, 0, 0, 0],[0, 0.35, 0, 0, 0, 0]]
        # inertia tensor of link with respect to
                # center of mass I = [L_xx, L_yy, L_zz,
                # L_xy, L_yz, L_xz]
        Tc=[[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]
        qlim=[[-160*deg, 160*deg],[-160*deg, 160*deg] ,[-160*deg, 160*deg] ,[-160*deg, 160*deg] ,[-160*deg, 160*deg] ,[-160*deg, 160*deg] ]    # minimum and maximum joint angle

        links = [
            RevoluteDH(
                d=base,       # link length (Dennavit-Hartenberg notation)
                a=0,          # link offset (Dennavit-Hartenberg notation)
                alpha=pi/2,   # link twist (Dennavit-Hartenberg notation)
                I=[0, 0.35, 0, 0, 0, 0],
                # inertia tensor of link with respect to
                # center of mass I = [L_xx, L_yy, L_zz,
                # L_xy, L_yz, L_xz]
                r=[0, 0, 0],
                # distance of ith origin to center of mass [x,y,z]
                # in link reference frame
                m=0,          # mass of link
                Jm=200e-6,    # actuator inertia
                G=-62.6111,   # gear ratio
                B=1.48e-3,    # actuator viscous friction coefficient (measured
                            # at the motor)
                Tc=[0.395, -0.435],
                # actuator Coulomb friction coefficient for
                # direction [-,+] (measured at the motor)
                qlim=[-160*deg, 160*deg]    # minimum and maximum joint angle
            ),

            RevoluteDH(
                d=0, a=0.4318, alpha=zero,
                I=[0.13, 0.524, 0.539, 0, 0, 0],
                r=[-0.3638, 0.006, 0.2275],
                m=17.4,
                Jm=200e-6,
                G=107.815,
                B=.817e-3,
                Tc=[0.126, -0.071],
                qlim=[-110*deg, 110*deg],  # qlim=[-45*deg, 225*deg]
            ),

            RevoluteDH(
                d=0.15005, a=0.0203, alpha=-pi/2,
                I=[0.066, 0.086, 0.0125, 0, 0, 0],
                r=[-0.0203, -0.0141, 0.070],
                m=4.8,
                Jm=200e-6,
                G=-53.7063,
                B=1.38e-3,
                Tc=[0.132, -0.105],
                qlim=[-135*deg, 135*deg]  # qlim=[-225*deg, 45*deg]
            ),

            RevoluteDH(
                d=0.4318, a=0, alpha=pi/2,
                I=[1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0],
                r=[0, 0.019, 0],
                m=0.82,
                Jm=33e-6,
                G=76.0364,
                B=71.2e-6,
                Tc=[11.2e-3, -16.9e-3],
                qlim=[-266*deg, 266*deg]  # qlim=[-110*deg, 170*deg]
            ),

            RevoluteDH(
                d=0, a=0, alpha=-pi/2,
                I=[0.3e-3, 0.4e-3, 0.3e-3, 0, 0, 0],
                r=[0, 0, 0],
                m=0.34,
                Jm=33e-6,
                G=71.923,
                B=82.6e-6,
                Tc=[9.26e-3, -14.5e-3],
                qlim=[-100*deg, 100*deg]
            ),

            RevoluteDH(
                d=0, a=0, alpha=zero,
                I=[0.15e-3, 0.15e-3, 0.04e-3, 0, 0, 0],
                r=[0, 0, 0.032],
                m=0.09,
                Jm=33e-6,
                G=76.686,
                B=36.7e-6,
                Tc=[3.96e-3, -10.5e-3],
                qlim=[-266*deg, 266*deg]
            )
        ]

        # for j in range(6):
        #     link = RevoluteDH(
        #         d=d[j],
        #         a=a[j],
        #         alpha=alpha[j],
        #         m=mass[j],
        #         r=center_of_mass[j],
        #         G=1
        #     )
        #     links.append(link)
    
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
        self.addconfiguration("qn", np.array([0, 0, 0, 0, 0, 0]))
if __name__ == '__main__':    # pragma nocover

    teco = TECOARM1(symbolic=False)
    print(teco)
    print(teco.dynamics())
