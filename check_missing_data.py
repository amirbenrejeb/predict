#!/usr/bin/env python3
"""
Simple script to check the missing_time.txt file and show statistics
"""

def check_missing_data_file():
    """Check and display contents of missing_time.txt file"""
    try:
        with open('missing_time.txt', 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print("No missing data entries found.")
            return
        
        print(f"Found {len(lines)} journeys with missing client-to-client data:")
        print("=" * 80)
        
        # Group by missing segment patterns
        missing_patterns = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract the pattern (everything after the first colon)
            if ':' in line:
                journey_id = line.split(':')[0].strip()
                pattern = ':'.join(line.split(':')[1:]).strip()
                
                if pattern not in missing_patterns:
                    missing_patterns[pattern] = []
                missing_patterns[pattern].append(journey_id)
        
        # Display patterns and counts
        for pattern, journey_ids in missing_patterns.items():
            print(f"\nPattern: {pattern}")
            print(f"Count: {len(journey_ids)} journeys")
            print(f"Journey IDs: {', '.join(journey_ids[:10])}")  # Show first 10
            if len(journey_ids) > 10:
                print(f"... and {len(journey_ids) - 10} more")
        
        print("\n" + "=" * 80)
        print(f"Total journeys with missing data: {len(lines)}")
        print(f"Unique missing patterns: {len(missing_patterns)}")
        
    except FileNotFoundError:
        print("missing_time.txt file not found. No missing data has been logged yet.")
    except Exception as e:
        print(f"Error reading missing_time.txt: {e}")

if __name__ == "__main__":
    check_missing_data_file()
