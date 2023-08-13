using Plots
using HDF5

file = "rl_environments_pendulum_sac_learning_curves.h5"

f = h5open(file, "r")


plt = plot(legend=false)
for (run_i, run) in enumerate(f)
    plot!(run["episode_returns"][:], label="run $run_i", linecolor = :grey, linewidth=2)
end
title!(plt, file)
xlabel!(plt, "Evaluation #")
ylabel!(plt, "Return")
display(plt)

close(f)