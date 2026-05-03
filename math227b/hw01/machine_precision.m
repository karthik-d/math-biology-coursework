% double-precision floating point numbers.
eps_machine_double = 1.0;
rnd_num_double = double(1.0);
while rnd_num_double + eps_machine_double > rnd_num_double
    eps_machine_double = eps_machine_double / 2;
end
eps_machine_double = eps_machine_double * 2;

disp("for double-precision floating point number:");
disp(eps_machine_double);


% single-precision floating point numbers.
rnd_num_single = single(1.0);
eps_machine_single = 1.0;
while rnd_num_single + eps_machine_single > rnd_num_single
    eps_machine_single = eps_machine_single / 2;
end
eps_machine_single = eps_machine_single * 2;

disp("for single-precision floating point number:");
disp(eps_machine_single);