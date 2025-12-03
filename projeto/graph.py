import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("traj_video2.csv")

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Trajetória no mundo
axs[0, 0].plot(df['X_world'], df['Y_world'], '-o')
axs[0, 0].invert_yaxis()
axs[0, 0].set_title("Trajetória no mundo")
axs[0, 0].set_xlabel("X (m)")
axs[0, 0].set_ylabel("Y (m)")
axs[0, 0].axis('equal')
axs[0, 0].grid(True)

# Posição X e Y no mundo ao longo do tempo
axs[0, 1].plot(df['frame'], df['X_world'], label='X_world')
axs[0, 1].plot(df['frame'], df['Y_world'], label='Y_world')
axs[0, 1].set_title("Posição no mundo")
axs[0, 1].set_xlabel("Frame")
axs[0, 1].set_ylabel("Posição (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Orientação (yaw) no mundo ao longo do tempo
axs[0, 2].plot(df['frame'], df['yaw_deg'])
axs[0, 2].set_title("Orientação (Yaw) no mundo")
axs[0, 2].set_xlabel("Frame")
axs[0, 2].set_ylabel("Yaw (graus)")
axs[0, 2].grid(True)

# Trajetória no plano da imagem
axs[1, 0].plot(df['u'], df['v'], '-o')
axs[1, 0].set_title("Trajetória no plano da imagem")
axs[1, 0].set_xlabel("u (pixels)")
axs[1, 0].set_ylabel("v (pixels)")
axs[1, 0].axis('equal')
axs[1, 0].grid(True)

# Posição u e v ao longo do tempo
axs[1, 1].plot(df['frame'], df['u'], label='u')
axs[1, 1].plot(df['frame'], df['v'], label='v')
axs[1, 1].set_title("Posição no plano da imagem")
axs[1, 1].set_xlabel("Frame")
axs[1, 1].set_ylabel("Pixels")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Área ao longo do tempo (opcional, pode remover se não quiser)
axs[1, 2].plot(df['frame'], df['area'])
axs[1, 2].set_title("Área detectada ao longo do tempo")
axs[1, 2].set_xlabel("Frame")
axs[1, 2].set_ylabel("Área")
axs[1, 2].grid(True)

plt.tight_layout()
plt.show()
