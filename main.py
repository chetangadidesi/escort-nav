import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from map import MAP_CONFIG

from simulation import EscortSimulation
from visualizer import DashboardVisualizer

def animate_escort(config):
    print("Initializing Simulation Brain...")
    sim = EscortSimulation(config)
    
    print("Initializing Dashboard Eyes...")
    vis = DashboardVisualizer(sim)

    def update(frame_idx_ignored):
        # 1. Think (Math & Logic)
        sim.step()
        
        # 2. Draw (Matplotlib)
        vis.render()

    anim = FuncAnimation(vis.fig_map, update, frames=5000, interval=10, repeat=False, blit=False)

    # Key Press Event hooked to the main map window
    def on_key(event):
        if event.key == ' ':
            sim.state['vip_paused'] = not sim.state['vip_paused']
            status = "PAUSED" if sim.state['vip_paused'] else "RESUMED"
            print(f"VIP Movement {status}. Swarm still active.")
            
    vis.fig_map.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
    return anim

if __name__ == "__main__":
    anim = animate_escort(MAP_CONFIG)