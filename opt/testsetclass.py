import random
import subprocess

class TestsetClass:
    def run_dftd4(self,  a1, a2, coord_path, molecule_charge):
        #print(molecule_charge)
        #print(coord_path)
        command_s6 = f"/homes2/chemie/wodynski/bin/panda6/bin/dftd4 --charge {molecule_charge} --input coord --param 1.0 0.0 {a1} {a2} --mbdscale 0.0 --silent {coord_path} | grep 'Dispersion energy:' | awk '{{print $3}}'"
        command_s8 = f"/homes2/chemie/wodynski/bin/panda6/bin/dftd4 --charge {molecule_charge} --input coord --param 0.0 1.0 {a1} {a2} --mbdscale 0.0 --silent {coord_path} | grep 'Dispersion energy:' | awk '{{print $3}}'"
        command_s9 = f"/homes2/chemie/wodynski/bin/panda6/bin/dftd4 --charge {molecule_charge} --input coord --param 0.0 0.0 {a1} {a2} --mbdscale 1.0 --silent {coord_path} | grep 'Dispersion energy:' | awk '{{print $3}}'"
        # Uruchomienie polecenia w powłoce i przechwycenie wyjścia
        result_s6 = subprocess.run(command_s6, shell=True, text=True, capture_output=True)
        result_s8 = subprocess.run(command_s8, shell=True, text=True, capture_output=True)
        result_s9 = subprocess.run(command_s9, shell=True, text=True, capture_output=True)

        if result_s6.returncode == 0 and result_s8.returncode == 0 and result_s9.returncode == 0:
            return {'s6': float(result_s6.stdout.strip()),
                    's8': float(result_s8.stdout.strip()),
                    's9': float(result_s9.stdout.strip())}
        else:
            print("Error running command:", result.stderr)
            return None

    def calculate_dispersion_energies(self,testset_name, a1, a2, atoms=False):
        self.dispersion_energy_results = {}
        path='/projects/kaupp/testsets/TestsetLibrary_artur/GMTKN55'
        if not atoms:
            for molecule, charge in self.charges.items():
                coord_path = f"{path}/{testset_name}/{molecule}/coord"
                energy = self.run_dftd4( a1, a2, coord_path, charge)
                if energy is not None:
                    self.dispersion_energy_results[molecule] = energy
                else:
                    print(f"Error calculating energy for {molecule}.")
        else:
            for molecule in self.molecules:
                self.dispersion_energy_results[molecule] = {'s6': 0.0, 's8': 0.0, 's9': 0.0}

