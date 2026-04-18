import numpy as np
import pygame
import matplotlib.pyplot as plt
import threading
import time

# ===================== Part 1: Setup, Physics, and Plotting =====================

# Simulation constants
WIDTH, HEIGHT = 800, 600
NUM_PARTICLES = 100
G = 1
SOFTENING = 1.0
DT = 0.1
FPS = 60

def run_simulation(name, ang_mom_config, run_time, stop_event):
    print(f"Running {name}...")

    # Initialize particle masses and positions
    np.random.seed(42)
    masses = np.random.rand(NUM_PARTICLES) * 5 + 1
    spread = 80
    center = np.array([WIDTH / 2, HEIGHT / 2])
    positions = center + np.random.randn(NUM_PARTICLES, 2) * spread

    # Assign initial velocities based on angular momentum configuration
    velocities = np.zeros_like(positions)
    for i in range(NUM_PARTICLES):
        rel = positions[i] - center
        perp = np.array([-rel[1], rel[0]])  # Perpendicular vector
        norm = np.linalg.norm(perp)
        if norm > 0:
            perp /= norm
        velocities[i] = ang_mom_config(rel, perp, i)

    total_energy_list = []
    angular_momentum_list = []
    time_list = []

    # Compute gravitational forces between all particles
    def compute_forces(pos, mass):
        force = np.zeros_like(pos)
        potential_energy = 0
        for i in range(len(pos)):
            dx = pos[i] - pos
            dist_sq = np.sum(dx ** 2, axis=1) + SOFTENING
            inv_dist3 = dist_sq ** -1.5
            f = -G * mass[i] * mass[:, None] * dx * inv_dist3[:, None]
            f[i] = 0
            force[i] = np.sum(f, axis=0)
            potential_energy += -0.5 * G * np.sum(mass[i] * mass * dist_sq ** -0.5)
        return force, potential_energy

    # Euler integration to update positions and velocities
    def update(pos, vel, mass):
        forces, pe = compute_forces(pos, mass)
        vel += forces / mass[:, None] * DT
        pos += vel * DT
        ke = 0.5 * np.sum(mass * np.sum(vel ** 2, axis=1))
        ang_mom = np.sum(np.cross(pos - center, mass[:, None] * vel))
        return pos, vel, ke + pe, ang_mom

    # Background thread for plotting energy and angular momentum
    def plot_thread():
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
        while not stop_event.is_set():
            if len(time_list) > 0:
                ax1.clear()
                ax2.clear()
                ax1.plot(time_list, total_energy_list, 'b-', label="Total Energy")
                ax2.plot(time_list, angular_momentum_list, 'g-', label="Angular Momentum")
                ax1.set_ylabel("Energy")
                ax2.set_ylabel("Angular Momentum")
                ax2.set_xlabel("Time")
                ax1.legend()
                ax2.legend()
                plt.pause(0.001)
            time.sleep(0.1)
        plt.close(fig)

    # Start the plot thread
    stop_event.clear()
    plot_thread_obj = threading.Thread(target=plot_thread)
    plot_thread_obj.start()

    # ===================== Part 2: Pygame Window and Main Loop =====================

    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Galaxy Simulation - {name}")
    clock = pygame.time.Clock()

    frame = 0
    sim_running = True

    while sim_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_running = False
                stop_event.set()

        # Update physics and track energy and momentum
        positions, velocities, E, L = update(positions, velocities, masses)
        total_energy_list.append(E)
        angular_momentum_list.append(L)
        time_list.append(frame * DT)

        # Draw particles
        win.fill((0, 0, 0))
        for pos in positions:
            x, y = pos
            if 0 <= x <= WIDTH and 0 <= y <= HEIGHT:
                pygame.draw.circle(win, (0, 255, 255), (int(x), int(y)), 3)
        pygame.display.flip()

        clock.tick(FPS)
        frame += 1

        if frame >= FPS * run_time or not sim_running:
            sim_running = False
            stop_event.set()

    plot_thread_obj.join()
    pygame.quit()
    print(f"{name} completed!\n")

# ===================== Part 3: Configurations and Execution =====================

# Different angular momentum configurations for galaxy types
def high_ang_mom(rel, perp, i):
    return 1.5 * perp  # Fast rotation – spiral structure

def low_ang_mom(rel, perp, i):
    return 0.3 * perp  # Slow rotation – elliptical structure

def intermediate_asymmetry(rel, perp, i):
    factor = 0.9 + 0.4 * np.sin(i)  # Add variation
    perp = perp + 0.3 * np.random.randn(*perp.shape)  # Add noise
    return factor * perp  # Irregular motion

# Main execution
if __name__ == "__main__":
    ORIGINAL_RUN_TIME = 20
    CASE_RUN_TIME = 10
    stop_event = threading.Event()

    # Run base and 3 comparative galaxy formation cases
    run_simulation("Original", high_ang_mom, ORIGINAL_RUN_TIME, stop_event)
    time.sleep(0.5)

    run_simulation("Case A: High angular momentum (Spiral)", high_ang_mom, CASE_RUN_TIME, stop_event)
    time.sleep(0.5)

    run_simulation("Case B: Low angular momentum (Elliptical)", low_ang_mom, CASE_RUN_TIME, stop_event)
    time.sleep(0.5)

    run_simulation("Case C: Intermediate with asymmetry (Irregular)", intermediate_asymmetry, CASE_RUN_TIME, stop_event)
    time.sleep(0.5)

    print("All runs completed!")