"""
auditor.py
----------
FairnessAuditor: records job completion events and computes
per-type fairness metrics for the factory scheduling system.

Every completed job is logged with its timing, energy, and
fairness-relevant attributes. FairnessReporter reads this log
to compute the five bias metrics.

Key attribute names on the Job object (from env/job_generator.py):
    job.job_type          'A', 'B', or 'C'
    job.priority          'normal' or 'high'
    job.completion_time   int  (set when job finishes in factory_gym.py)
    job.start_time        int  (set when job is assigned; None if reset)
    job.energy_cost       float
    job.processing_time   int
    job.assigned_machine  int  (set on assignment; None if paused by breakdown)
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


class FairnessAuditor:
    """
    Records and aggregates scheduling outcomes across job types and priorities.

    Usage pattern inside the evaluation loop:

        auditor = FairnessAuditor()

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            auditor.new_episode()

            while True:
                count_before = len(env.completed_jobs)       # snapshot
                obs, rewards, terms, truncs, _ = env.step(actions)

                for job in env.completed_jobs[count_before:]: # newly done
                    auditor.record_job_completion(job, job.assigned_machine or 0)

                if terms['__all__'] or truncs['__all__']:
                    break

        auditor.save_to_csv('logs/fairness_audit.csv')
    """

    # Column names for the audit log DataFrame
    COLUMNS = [
        'episode', 'job_id', 'type', 'priority',
        'arrival', 'deadline', 'processing_time',
        'completion', 'tardiness', 'wait_time',
        'machine', 'energy_consumed',
    ]

    def __init__(self):
        self.log: List[Dict[str, Any]] = []
        self.episode_id: int = 0

    # ──────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────────────

    def new_episode(self) -> None:
        """Call once at the start of each evaluation episode."""
        self.episode_id += 1

    def record_job_completion(
        self,
        job,            # Job object from env.job_generator
        machine_id: int,
    ) -> None:
        """
        Record one completed job's outcomes for fairness analysis.

        Args:
            job:        Completed Job object. job.completion_time must be set.
            machine_id: ID of machine that processed the job (0, 1, or 2).

        Silently skips if job.completion_time is None (not yet complete).
        """
        if job.completion_time is None:
            return  # safety guard — should not happen for completed jobs

        tardiness = max(0, job.completion_time - job.deadline)

        # wait_time = steps between arrival and processing start
        # start_time can be None if the job was paused by a breakdown and
        # reset_assignment() was called; use 0 as conservative fallback.
        if job.start_time is not None:
            wait_time = max(0, job.start_time - job.arrival_time)
        else:
            wait_time = 0

        # Total energy = cost-per-step × number of processing steps
        energy_consumed = round(job.energy_cost * job.processing_time, 4)

        self.log.append({
            'episode':         self.episode_id,
            'job_id':          job.job_id,
            'type':            job.job_type,    # 'A', 'B', or 'C'
            'priority':        job.priority,    # 'normal' or 'high'
            'arrival':         job.arrival_time,
            'deadline':        job.deadline,
            'processing_time': job.processing_time,
            'completion':      job.completion_time,
            'tardiness':       tardiness,
            'wait_time':       wait_time,
            'machine':         machine_id,
            'energy_consumed': energy_consumed,
        })

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute per-type aggregate metrics from all recorded jobs.

        Returns a dict with per-type count, avg_tardiness, tardiness_rate,
        avg_wait_time, and total_energy. Returns {} when no jobs logged.
        """
        if not self.log:
            return {}

        df = self.get_log_df()
        result: Dict[str, Any] = {}

        for job_type in ['A', 'B', 'C']:
            subset = df[df['type'] == job_type]
            if len(subset) == 0:
                result[job_type] = {
                    'count':          0,
                    'avg_tardiness':  0.0,
                    'tardiness_rate': 0.0,
                    'avg_wait_time':  0.0,
                    'total_energy':   0.0,
                }
            else:
                n_tardy = int((subset['tardiness'] > 0).sum())
                result[job_type] = {
                    'count':          int(len(subset)),
                    'avg_tardiness':  float(subset['tardiness'].mean()),
                    'tardiness_rate': float(n_tardy / len(subset)),
                    'avg_wait_time':  float(subset['wait_time'].mean()),
                    'total_energy':   float(subset['energy_consumed'].sum()),
                }

        # Priority breakdown (normal vs high/rush)
        for priority in ['normal', 'high']:
            subset = df[df['priority'] == priority]
            key = f'priority_{priority}'
            if len(subset) > 0:
                result[key] = {
                    'count':          int(len(subset)),
                    'avg_tardiness':  float(subset['tardiness'].mean()),
                    'tardiness_rate': float((subset['tardiness'] > 0).mean()),
                }
            else:
                result[key] = {'count': 0, 'avg_tardiness': 0.0, 'tardiness_rate': 0.0}

        return result

    def get_log_df(self) -> pd.DataFrame:
        """Return the full audit log as a pandas DataFrame."""
        if not self.log:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.DataFrame(self.log)

    def save_to_csv(self, path: str = 'logs/fairness_audit.csv') -> None:
        """Save the complete audit log to CSV. Creates the directory if needed."""
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        df = self.get_log_df()
        df.to_csv(path, index=False)
        print(f"[FairnessAuditor] {len(df)} job records saved → {path}")

    def total_jobs_logged(self) -> int:
        """Return total number of completed jobs logged across all episodes."""
        return len(self.log)

    def reset(self) -> None:
        """Clear all logged data. Episode counter is preserved."""
        self.log = []

    def __repr__(self) -> str:
        return (
            f"FairnessAuditor(episodes={self.episode_id}, "
            f"jobs_logged={len(self.log)})"
        )