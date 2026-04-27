"""Generate synthetic case histories based on Norwegian homicide research."""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any


class CaseGenerator:
    """Generate realistic synthetic case histories."""
    
    def __init__(self, seed=42):
        random.seed(seed)
        self.case_counter = 0
    
    def generate_high_risk_case(self) -> Dict[str, Any]:
        """Generate case with escalating violence pattern (later escalated)."""
        self.case_counter += 1
        case_id = f"SYNTH_HIGH_{self.case_counter:03d}"
        
        start_date = datetime(2022, 1, 1)
        events = []
        
        # Escalating pattern
        events.append({
            'date': (start_date + timedelta(days=15)).isoformat(),
            'type': 'police_report',
            'violence': True,
            'severity': 3
        })
        
        events.append({
            'date': (start_date + timedelta(days=35)).isoformat(),
            'type': 'threat',
            'severity': 6,
            'details': 'Explicit threats recorded'
        })
        
        events.append({
            'date': (start_date + timedelta(days=60)).isoformat(),
            'type': 'police_report',
            'violence': True,
            'severity': 5,
            'details': 'Escalated violence'
        })
        
        events.append({
            'date': (start_date + timedelta(days=85)).isoformat(),
            'type': 'restraining_order_breach',
            'details': 'Violated order'
        })
        
        events.append({
            'date': (start_date + timedelta(days=110)).isoformat(),
            'type': 'threat',
            'severity': 8,
            'weapon_mention': True,
            'details': 'Serious threat with weapon mention'
        })
        
        events.append({
            'date': (start_date + timedelta(days=135)).isoformat(),
            'type': 'isolation_behavior',
            'details': 'Controlling behavior, limiting contact'
        })
        
        # Add barnevern contact
        events.append({
            'date': (start_date + timedelta(days=150)).isoformat(),
            'type': 'child_welfare_contact',
            'details': 'Report to barnevern'
        })
        
        return {
            'case_id': case_id,
            'start_date': start_date.isoformat(),
            'events': events,
            'outcome': 'escalated_to_lethal',
            'risk_category': 'high'
        }
    
    def generate_low_risk_case(self) -> Dict[str, Any]:
        """Generate isolated incident case (resolved safely)."""
        self.case_counter += 1
        case_id = f"SYNTH_LOW_{self.case_counter:03d}"
        
        start_date = datetime(2022, 3, 1)
        events = []
        
        # Single incident, no escalation
        events.append({
            'date': (start_date + timedelta(days=10)).isoformat(),
            'type': 'police_report',
            'violence': False,
            'severity': 1,
            'details': 'Isolated argument'
        })
        
        # Resolution
        events.append({
            'date': (start_date + timedelta(days=30)).isoformat(),
            'type': 'case_closed',
            'details': 'No further incidents'
        })
        
        return {
            'case_id': case_id,
            'start_date': start_date.isoformat(),
            'events': events,
            'outcome': 'resolved',
            'risk_category': 'low'
        }
    
    def generate_dataset(self, n_cases: int = 500, high_risk_ratio: float = 0.3) -> List[Dict]:
        """Generate full dataset."""
        cases = []
        n_high = int(n_cases * high_risk_ratio)
        n_low = n_cases - n_high
        
        for _ in range(n_high):
            cases.append(self.generate_high_risk_case())
        
        for _ in range(n_low):
            cases.append(self.generate_low_risk_case())
        
        return cases


if __name__ == "__main__":
    # Generate example dataset
    gen = CaseGenerator(seed=42)
    dataset = gen.generate_dataset(n_cases=500)
    
    # Save to file
    with open('synthetic_cases.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} cases")
    print(f"High-risk cases: {len([c for c in dataset if c['risk_category'] == 'high'])}")
    print(f"Low-risk cases: {len([c for c in dataset if c['risk_category'] == 'low'])}")
