"""
High Harmonic Generation (HHG) Simulation for Solids
==================================================

This simulation models HHG in solid materials using:
1. Semiconductor Bloch Equations (SBE)
2. Strong Field Approximation
3. Time-dependent electric field interaction
4. Band structure effects

Based on current research in solid-state HHG physics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
import matplotlib.patches as patches

# Physical constants
hbar = 1.054571817e-34  # J⋅s
e = 1.602176634e-19     # C
m_e = 9.1093837015e-31  # kg
c = 299792458           # m/s
epsilon_0 = 8.8541878128e-12  # F/m

class HHGSolidSimulation:
    def __init__(self, material_params=None):
        """
        Initialize HHG simulation for solid materials.
        
        Parameters:
        -----------
        material_params : dict
            Material parameters (bandgap, effective masses, etc.)
        """
        # GaSe material parameters (layered semiconductor)
        if material_params is None:
            self.material = {
                'bandgap': 2.0 * e,  # eV to Joules (indirect bandgap ~2.0 eV)
                'direct_bandgap': 2.1 * e,  # Direct bandgap slightly higher
                'lattice_constant_a': 3.75e-10,  # in-plane lattice constant (meters)
                'lattice_constant_c': 15.9e-10,  # out-of-plane (layer separation)
                'effective_mass_e_parallel': 0.14 * m_e,  # electron mass parallel to layers
                'effective_mass_e_perp': 0.9 * m_e,      # electron mass perpendicular
                'effective_mass_h_parallel': 0.35 * m_e,  # hole mass parallel to layers
                'effective_mass_h_perp': 1.2 * m_e,      # hole mass perpendicular
                'dipole_moment': 2.5e-29,  # C⋅m (enhanced due to layered structure)
                'dephasing_time': 8e-15,  # seconds (longer due to reduced scattering)
                'anisotropy_factor': 6.4,  # optical anisotropy (c/a axis ratio)
                'nonlinear_susceptibility': 3.2e-21,  # m²/V² (strong nonlinearity)
                'excitonic_binding': 0.02 * e,  # eV (weak excitonic effects)
                'name': 'GaSe'
            }
        else:
            self.material = material_params
            
        # Laser parameters optimized for GaSe HHG experiments
        self.laser = {
            'wavelength': 1030e-9,  # meters (common for GaSe experiments)
            'frequency': c / (1030e-9),  # Hz
            'intensity': 5e12,  # W/cm² (lower intensity due to GaSe's high nonlinearity)
            'pulse_duration': 35e-15,  # seconds (slightly longer pulses)
            'cep': 0,  # carrier envelope phase
            'polarization': 'linear'  # linear polarization (important for layered materials)
        }
        
        # Simulation parameters
        self.time_span = 200e-15  # seconds
        self.dt = 0.1e-15  # time step
        self.t = np.arange(0, self.time_span, self.dt)
        
        # Initialize state variables
        self.reset_simulation()
        
    def reset_simulation(self):
        """Reset simulation state variables."""
        self.n_cb = np.zeros(len(self.t))  # conduction band population
        self.n_vb = np.ones(len(self.t))   # valence band population
        self.polarization = np.zeros(len(self.t), dtype=complex)
        self.current = np.zeros(len(self.t))
        self.electric_field = np.zeros(len(self.t))
        
    def laser_field(self, time):
        """
        Calculate the time-dependent laser electric field.
        
        Parameters:
        -----------
        time : float or array
            Time in seconds
            
        Returns:
        --------
        E_field : float or array
            Electric field amplitude in V/m
        """
        omega = 2 * np.pi * self.laser['frequency']
        tau = self.laser['pulse_duration']
        I0 = self.laser['intensity'] * 1e4  # Convert W/cm² to W/m²
        
        # Peak electric field amplitude
        E0 = np.sqrt(2 * I0 / (c * epsilon_0))
        
        # Gaussian envelope with sine carrier
        envelope = np.exp(-2 * np.log(2) * (time - self.time_span/2)**2 / tau**2)
        carrier = np.sin(omega * time + self.laser['cep'])
        
        return E0 * envelope * carrier
    
    def bloch_equations(self, t, y):
        """
        Enhanced Semiconductor Bloch equations for GaSe HHG.
        Includes anisotropy and excitonic effects specific to layered materials.
        
        Parameters:
        -----------
        t : float
            Current time
        y : array
            State vector [Re(p), Im(p), n_cv]
            where p is polarization and n_cv is population difference
            
        Returns:
        --------
        dydt : array
            Time derivatives of state variables
        """
        # Extract state variables
        p_real, p_imag, n_cv = y
        p_complex = p_real + 1j * p_imag
        
        # Current electric field
        E_field = self.laser_field(t)
        
        # GaSe-specific material parameters
        omega_gap = self.material['bandgap'] / hbar
        omega_direct = self.material['direct_bandgap'] / hbar
        gamma = 1 / self.material['dephasing_time']
        d_cv = self.material['dipole_moment']
        
        # Excitonic enhancement factor for GaSe
        excitonic_energy = self.material['excitonic_binding'] / hbar
        
        # Enhanced dipole moment due to excitonic effects
        d_eff = d_cv * (1 + 0.1 * np.exp(-abs(n_cv)))  # Field-dependent enhancement
        
        # Anisotropy factor (for in-plane vs out-of-plane response)
        anisotropy = self.material['anisotropy_factor']
        
        # Modified Bloch equations for GaSe
        # Polarization dynamics with excitonic corrections
        dp_dt = (-1j * (omega_gap - excitonic_energy) * p_complex - 
                gamma * p_complex + 
                1j * (d_eff * E_field / hbar) * n_cv * anisotropy)
        
        # Population dynamics with nonlinear corrections
        dn_cv_dt = (-2 * (d_eff * E_field / hbar) * p_imag - 
                   0.1 * n_cv * (1 + n_cv**2))  # Auger-like recombination
        
        return [dp_dt.real, dp_dt.imag, dn_cv_dt]
    
    def run_simulation(self):
        """Run the full HHG simulation."""
        print(f"Running HHG simulation for {self.material['name']}")
        print(f"Laser intensity: {self.laser['intensity']:.1e} W/cm²")
        print(f"Simulation time: {self.time_span*1e15:.1f} fs")
        
        # Initial conditions [Re(p), Im(p), n_cv]
        y0 = [0.0, 0.0, -1.0]  # Ground state (valence band full)
        
        # Solve Bloch equations
        sol = solve_ivp(self.bloch_equations, [0, self.time_span], y0, 
                       t_eval=self.t, method='RK45', rtol=1e-8)
        
        if not sol.success:
            print("Warning: ODE solver did not converge properly")
        
        # Extract results
        self.polarization = sol.y[0] + 1j * sol.y[1]
        n_cv = sol.y[2]
        
        # Calculate physical quantities
        self.electric_field = np.array([self.laser_field(time) for time in self.t])
        
        # Current density (time derivative of polarization)
        self.current = np.gradient(self.polarization.real, self.dt)
        
        # Conduction band population
        self.n_cb = (1 + n_cv) / 2
        self.n_vb = (1 - n_cv) / 2
        
        print("Simulation completed successfully!")
        
    def calculate_harmonic_spectrum(self):
        """Calculate the HHG spectrum from the current."""
        # Window function to reduce spectral artifacts
        window = np.hanning(len(self.current))
        windowed_current = self.current * window
        
        # FFT to get frequency spectrum
        current_fft = fft(windowed_current)
        frequencies = fftfreq(len(self.t), self.dt)
        
        # Convert to positive frequencies only
        positive_freq_mask = frequencies > 0
        frequencies = frequencies[positive_freq_mask]
        spectrum = np.abs(current_fft[positive_freq_mask])**2
        
        # Normalize spectrum
        spectrum = spectrum / np.max(spectrum)
        
        # Convert to harmonic orders
        fundamental_freq = self.laser['frequency']
        harmonic_orders = frequencies / fundamental_freq
        
        return harmonic_orders, spectrum
    
    def plot_results(self):
        """Create comprehensive plots of HHG simulation results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'High Harmonic Generation in {self.material["name"]} (Layered Semiconductor)', 
                    fontsize=16, fontweight='bold')
        
        # Time axis in femtoseconds
        t_fs = self.t * 1e15
        
        # Plot 1: Electric field and polarization
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(t_fs, self.electric_field * 1e-9, 'b-', linewidth=2, 
                        label='Electric Field')
        line2 = ax1_twin.plot(t_fs, self.polarization.real * 1e30, 'r-', linewidth=2, 
                             label='Polarization (Re)')
        
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Electric Field (GV/m)', color='b')
        ax1_twin.set_ylabel('Polarization (×10⁻³⁰ C⋅m)', color='r')
        ax1.set_title('Driving Field & Material Response')
        ax1.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # Plot 2: Band populations
        ax2 = axes[0, 1]
        ax2.plot(t_fs, self.n_vb, 'b-', linewidth=2, label='Valence Band')
        ax2.plot(t_fs, self.n_cb, 'r-', linewidth=2, label='Conduction Band')
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Population')
        ax2.set_title('Electronic Band Populations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot 3: Current density
        ax3 = axes[0, 2]
        ax3.plot(t_fs, self.current * 1e12, 'g-', linewidth=2)
        ax3.set_xlabel('Time (fs)')
        ax3.set_ylabel('Current Density (pA/m²)')
        ax3.set_title('Induced Current (HHG Source)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: HHG Spectrum
        harmonic_orders, spectrum = self.calculate_harmonic_spectrum()
        
        ax4 = axes[1, 0]
        # Plot only up to 50th harmonic for clarity
        mask = harmonic_orders <= 50
        ax4.semilogy(harmonic_orders[mask], spectrum[mask], 'r-', linewidth=2)
        ax4.set_xlabel('Harmonic Order')
        ax4.set_ylabel('Intensity (arb. units)')
        ax4.set_title('HHG Spectrum')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 50])
        
        # Add vertical lines for odd harmonics
        for i in range(1, 50, 2):
            if i <= 50:
                ax4.axvline(i, color='gray', alpha=0.3, linestyle='--')
        
        # Plot 5: Energy diagram
        ax5 = axes[1, 1]
        self.plot_energy_diagram(ax5)
        
        # Plot 6: Phase space (E-field vs Current)
        ax6 = axes[1, 2]
        ax6.plot(self.electric_field * 1e-9, self.current * 1e12, 'purple', 
                linewidth=2, alpha=0.7)
        ax6.set_xlabel('Electric Field (GV/m)')
        ax6.set_ylabel('Current Density (pA/m²)')
        ax6.set_title('Nonlinear Response (Hysteresis)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional detailed spectrum plot
        self.plot_detailed_spectrum()
        
    def plot_energy_diagram(self, ax):
        """Plot energy band diagram for GaSe with layered structure effects."""
        # Energy levels for GaSe
        E_vb = 0
        E_cb_indirect = self.material['bandgap'] / e  # Convert to eV
        E_cb_direct = self.material['direct_bandgap'] / e
        E_exc = self.material['excitonic_binding'] / e
        
        # Draw bands with GaSe-specific features
        band_width = 2
        
        # Valence band (with layer structure indication)
        vb_rect = patches.Rectangle((-1, E_vb - 0.5), band_width, 1, 
                                   linewidth=2, edgecolor='blue', facecolor='lightblue', 
                                   alpha=0.7, label='Valence Band')
        
        # Conduction band (showing indirect and direct gaps)
        cb_rect_indirect = patches.Rectangle((-1, E_cb_indirect - 0.5), band_width, 0.8, 
                                           linewidth=2, edgecolor='red', facecolor='lightcoral', 
                                           alpha=0.5, label='CB (indirect)')
        cb_rect_direct = patches.Rectangle((-0.5, E_cb_direct - 0.5), 1, 0.8, 
                                         linewidth=2, edgecolor='darkred', facecolor='red', 
                                         alpha=0.7, label='CB (direct)')
        
        ax.add_patch(vb_rect)
        ax.add_patch(cb_rect_indirect)
        ax.add_patch(cb_rect_direct)
        
        # Draw excitonic level
        exc_line = patches.Rectangle((-0.8, E_cb_indirect - E_exc - 0.1), 1.6, 0.2, 
                                   linewidth=1, edgecolor='green', facecolor='lightgreen', 
                                   alpha=0.8, label='Exciton')
        ax.add_patch(exc_line)
        
        # Draw transitions
        # Direct transition
        ax.arrow(0, E_vb + 0.5, 0, E_cb_direct - E_vb - 1, head_width=0.1, 
                head_length=0.15, fc='purple', ec='purple', linewidth=2)
        ax.text(0.2, (E_vb + E_cb_direct)/2, 'Direct\n(2.1 eV)', fontsize=10, 
               color='purple', fontweight='bold')
        
        # Indirect transition
        ax.arrow(-0.7, E_vb + 0.5, 0, E_cb_indirect - E_vb - 1, head_width=0.08, 
                head_length=0.12, fc='orange', ec='orange', linewidth=1.5)
        ax.text(-1.2, (E_vb + E_cb_indirect)/2, 'Indirect\n(2.0 eV)', fontsize=9, 
               color='orange', fontweight='bold')
        
        # Layer structure indication
        for i in range(3):
            y_layer = E_vb - 0.8 - i * 0.3
            ax.plot([-1.5, 1.5], [y_layer, y_layer], 'k-', linewidth=1, alpha=0.3)
        ax.text(1.2, E_vb - 1.2, 'Layered\nStructure', fontsize=8, ha='center', alpha=0.7)
        
        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([E_vb - 2, E_cb_direct + 1])
        ax.set_ylabel('Energy (eV)')
        ax.set_title('GaSe Band Structure')
        ax.set_xticks([])
        ax.legend(loc='upper left', fontsize=8)
        
    def plot_detailed_spectrum(self):
        """Plot detailed HHG spectrum with comparison to experimental features."""
        harmonic_orders, spectrum = self.calculate_harmonic_spectrum()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Linear scale spectrum
        mask = harmonic_orders <= 30
        ax1.plot(harmonic_orders[mask], spectrum[mask], 'r-', linewidth=2, 
                label='HHG Spectrum')
        ax1.set_xlabel('Harmonic Order')
        ax1.set_ylabel('Intensity (normalized)')
        ax1.set_title('HHG Spectrum - Linear Scale')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Mark odd harmonics
        for i in range(1, 30, 2):
            if i <= 30:
                ax1.axvline(i, color='gray', alpha=0.5, linestyle=':', linewidth=1)
                if i <= 21:  # Label first few harmonics
                    ax1.text(i, 0.9, f'H{i}', rotation=90, ha='center', 
                            fontsize=8, alpha=0.7)
        
        # Log scale spectrum (full range)
        mask_full = harmonic_orders <= 100
        ax2.semilogy(harmonic_orders[mask_full], spectrum[mask_full], 'b-', 
                    linewidth=2, label='HHG Spectrum (log scale)')
        ax2.set_xlabel('Harmonic Order')
        ax2.set_ylabel('Intensity (log scale)')
        ax2.set_title('HHG Spectrum - Log Scale (Extended Range)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add annotations for typical experimental features
        ax2.annotate('Perturbative regime\n(low harmonics)', 
                    xy=(5, 1e-2), xytext=(15, 1e-1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, ha='center')
        
        ax2.annotate('Plateau region\n(characteristic of HHG)', 
                    xy=(15, 1e-4), xytext=(25, 1e-3),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, ha='center')
        
        plt.tight_layout()
        plt.show()

def compare_materials():
    """Compare HHG in GaSe and other layered/2D materials."""
    materials = {
        'GaSe': {
            'bandgap': 2.0 * e,
            'direct_bandgap': 2.1 * e,
            'lattice_constant_a': 3.75e-10,
            'lattice_constant_c': 15.9e-10,
            'effective_mass_e_parallel': 0.14 * m_e,
            'effective_mass_e_perp': 0.9 * m_e,
            'effective_mass_h_parallel': 0.35 * m_e,
            'effective_mass_h_perp': 1.2 * m_e,
            'dipole_moment': 2.5e-29,
            'dephasing_time': 8e-15,
            'anisotropy_factor': 6.4,
            'nonlinear_susceptibility': 3.2e-21,
            'excitonic_binding': 0.02 * e,
            'name': 'GaSe'
        },
        'MoS2': {
            'bandgap': 1.8 * e,
            'direct_bandgap': 1.9 * e,
            'lattice_constant_a': 3.16e-10,
            'lattice_constant_c': 12.3e-10,
            'effective_mass_e_parallel': 0.35 * m_e,
            'effective_mass_e_perp': 0.6 * m_e,
            'effective_mass_h_parallel': 0.45 * m_e,
            'effective_mass_h_perp': 0.7 * m_e,
            'dipole_moment': 1.8e-29,
            'dephasing_time': 6e-15,
            'anisotropy_factor': 4.0,
            'nonlinear_susceptibility': 2.1e-21,
            'excitonic_binding': 0.5 * e,  # Strong excitonic effects
            'name': 'MoS2'
        },
        'GaS': {
            'bandgap': 2.5 * e,
            'direct_bandgap': 2.6 * e,
            'lattice_constant_a': 3.58e-10,
            'lattice_constant_c': 15.5e-10,
            'effective_mass_e_parallel': 0.15 * m_e,
            'effective_mass_e_perp': 0.8 * m_e,
            'effective_mass_h_parallel': 0.4 * m_e,
            'effective_mass_h_perp': 1.1 * m_e,
            'dipole_moment': 2.2e-29,
            'dephasing_time': 7e-15,
            'anisotropy_factor': 5.8,
            'nonlinear_susceptibility': 2.8e-21,
            'excitonic_binding': 0.03 * e,
            'name': 'GaS'
        }
    }
    
    plt.figure(figsize=(14, 10))
    colors = ['red', 'blue', 'green']
    linestyles = ['-', '--', '-.']
    
    # Main comparison plot
    plt.subplot(2, 2, 1)
    for i, (mat_name, mat_params) in enumerate(materials.items()):
        print(f"\nSimulating {mat_name}...")
        hhg_sim = HHGSolidSimulation(mat_params)
        hhg_sim.run_simulation()
        
        harmonic_orders, spectrum = hhg_sim.calculate_harmonic_spectrum()
        mask = harmonic_orders <= 40
        
        plt.semilogy(harmonic_orders[mask], spectrum[mask], 
                    color=colors[i], linewidth=2, linestyle=linestyles[i], 
                    label=f'{mat_name} (Eg={mat_params["bandgap"]/e:.1f} eV)')
    
    plt.xlabel('Harmonic Order')
    plt.ylabel('Intensity (arb. units)')
    plt.title('HHG in Layered Semiconductors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 40])
    
    # Bandgap comparison
    plt.subplot(2, 2, 2)
    bandgaps = [mat['bandgap']/e for mat in materials.values()]
    names = list(materials.keys())
    bars = plt.bar(names, bandgaps, color=colors)
    plt.ylabel('Bandgap (eV)')
    plt.title('Material Bandgaps')
    plt.xticks(rotation=45)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{bandgaps[i]:.1f}', ha='center', fontweight='bold')
    
    # Anisotropy comparison
    plt.subplot(2, 2, 3)
    anisotropies = [mat['anisotropy_factor'] for mat in materials.values()]
    bars = plt.bar(names, anisotropies, color=colors)
    plt.ylabel('Anisotropy Factor')
    plt.title('Optical Anisotropy')
    plt.xticks(rotation=45)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{anisotropies[i]:.1f}', ha='center', fontweight='bold')
    
    # Excitonic binding energy
    plt.subplot(2, 2, 4)
    exc_binding = [mat['excitonic_binding']/e*1000 for mat in materials.values()]  # in meV
    bars = plt.bar(names, exc_binding, color=colors)
    plt.ylabel('Excitonic Binding Energy (meV)')
    plt.title('Excitonic Effects')
    plt.xticks(rotation=45)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{exc_binding[i]:.0f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== High Harmonic Generation in GaSe (Layered Semiconductor) ===")
    print("Based on Enhanced Semiconductor Bloch Equations")
    print("Including excitonic effects and optical anisotropy\n")
    
    # Print GaSe material properties
    print("GaSe Material Properties:")
    print("- Indirect bandgap: 2.0 eV")
    print("- Direct bandgap: 2.1 eV") 
    print("- Layered structure with strong optical anisotropy")
    print("- Enhanced nonlinear optical response")
    print("- Optimized for 1030 nm driving wavelength\n")
    
    # Run main simulation for GaSe
    hhg_simulation = HHGSolidSimulation()
    hhg_simulation.run_simulation()
    hhg_simulation.plot_results()
    
    # Compare with other layered materials
    print("\n=== Comparison with Other Layered Materials ===")
    compare_materials()
    
    print("\nGaSe HHG simulation completed!")
    print("Key features to observe:")
    print("- Enhanced harmonic yield due to strong nonlinearity")
    print("- Anisotropic response from layered structure")  
    print("- Lower threshold intensity compared to bulk semiconductors")
    print("- Extended harmonic cutoff due to reduced effective mass")
