"""
Convenience entry point for running the self-improving research loop.
"""

from src.core.loop import SelfDrivingResearchLoop

if __name__ == "__main__":
    loop = SelfDrivingResearchLoop()
    # Runs the prioritized hypothesis backlog with fast 3000-sample validation
    loop.run_all_hypotheses(sample_size=3000)
