function [sig11, sig12, sig22]=cal_stress(F11,F12,F21,F22,nu,la)
J=F11.*F22-F12.*F21;
for i_s=1:length(F11)
    F=[F11(i_s) F12(i_s);
        F21(i_s) F22(i_s)];

    P=(nu*(F-inv(F)')+la*(J(i_s)-1)*J(i_s)*inv(F)');

    sig=1/J(i_s)*P*F';
    sig11(i_s,1)=sig(1,1);
    sig12(i_s,1)=sig(1,2);
    sig21(i_s,1)=sig(2,1);
    sig22(i_s,1)=sig(2,2);

    S=inv(F)*P;
    s11(i_s,1)=S(1,1);
    s12(i_s,1)=S(1,2);
    s21(i_s,1)=S(2,1);
    s22(i_s,1)=S(2,2);
end