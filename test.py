import matplotlib.pyplot as plt

# Create a figure and axes object
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting df_state_churn
ax.bar(df_state_churn['state'], df_state_churn['Count'], label='Churn')

# Plotting df_state_nchurn
ax.bar(df_state_nchurn['state'], df_state_nchurn['Count'], label='Non-Churn')

# Set labels and title
ax.set_xlabel('State')
ax.set_ylabel('Count')
ax.set_title('Churn vs. Non-Churn by State')

# Set legend
ax.legend()

# Display the plot
plt.show()
