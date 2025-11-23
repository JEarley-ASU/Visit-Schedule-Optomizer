import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

def generate_schedule(students_df, professor_df, time_slots, max_students=2):
    # Make copies to avoid modifying original data
    students_df = students_df.copy()
    professor_df = professor_df.copy()
    
    students_df.replace(' ', None, inplace=True)  # Replace empty spaces with None
    students = students_df.index.tolist()

    # Ensure professor_df has the correct columns assigned
    # Check if columns match, if not, try to map them
    if list(professor_df.columns) != time_slots:
        # If column count matches, just rename
        if len(professor_df.columns) == len(time_slots):
            professor_df.columns = time_slots
        else:
            # Column count mismatch - this is an error
            raise ValueError(f"Professor data has {len(professor_df.columns)} columns but {len(time_slots)} time slots expected. "
                           f"Columns: {list(professor_df.columns)}, Time slots: {time_slots}")
    
    # Ensure professor availability values are numeric (0 or 1)
    for col in professor_df.columns:
        professor_df[col] = pd.to_numeric(professor_df[col], errors='coerce').fillna(0).astype(int)
        # Ensure values are 0 or 1
        professor_df[col] = professor_df[col].apply(lambda x: 1 if x > 0 else 0)
    
    professors = professor_df.index.tolist()
    
    # Validate that we have students and professors
    if not students:
        raise ValueError("No students/visitors found in the data.")
    if not professors:
        raise ValueError("No professors found in the data.")
    
    # Check if any professors have at least one available slot
    prof_has_availability = False
    for prof in professors:
        if any(professor_df.loc[prof, slot] == 1 for slot in time_slots):
            prof_has_availability = True
            break
    
    if not prof_has_availability:
        raise ValueError("No professors have any available time slots. Please check professor availability.")

    # Set max meetings dynamically based on the number of time slots
    max_meetings = len(time_slots)

    # Create a dictionary to store preferences with weights (higher preference = higher weight)
    # Preferences are in columns like "Preference 1", "Preference 2", etc.
    preferences = {}
    
    # First pass: count preferences for each student to find max
    student_pref_counts = {}
    for student in students:
        student_row = students_df.loc[student]
        pref_columns = sorted([col for col in students_df.columns if col.startswith('Preference')], 
                            key=lambda x: int(x.split()[-1]))
        
        valid_count = 0
        for col in pref_columns:
            prof = student_row[col]
            if prof is not None and str(prof).strip() != '' and str(prof).strip() != ' ':
                prof = str(prof).strip()
                if prof in professors:
                    valid_count += 1
        
        student_pref_counts[student] = valid_count
    
    # Find maximum number of preferences any student has
    max_total_prefs = max(student_pref_counts.values()) if student_pref_counts else 1
    
    # Second pass: assign weights with priority boost for students with fewer preferences
    for student in students:
        student_prefs = {}
        # Get the row as a Series and iterate through preference columns in order
        student_row = students_df.loc[student]
        # Iterate through columns in order (they should be sorted: Preference 1, Preference 2, etc.)
        pref_columns = sorted([col for col in students_df.columns if col.startswith('Preference')], 
                            key=lambda x: int(x.split()[-1]))
        
        # Count valid preferences to set weights correctly
        valid_prefs = []
        for i, col in enumerate(pref_columns):
            prof = student_row[col]
            # Check if preference is valid (not None, not empty string, not just whitespace)
            if prof is not None and str(prof).strip() != '' and str(prof).strip() != ' ':
                prof = str(prof).strip()
                if prof in professors:
                    valid_prefs.append((prof, i))
        
        # Get number of preferences for this student
        num_prefs = len(valid_prefs)
        
        # Calculate priority multiplier: students with fewer preferences get exponentially higher priority
        # Use exponential multiplier to ensure significant difference
        # Formula: multiplier = 10^(max_total_prefs - num_prefs)
        # This means:
        # - Student with 1 preference (when max is 5): multiplier = 10^4 = 10,000
        # - Student with 3 preferences (when max is 5): multiplier = 10^2 = 100
        # - Student with 5 preferences (when max is 5): multiplier = 10^0 = 1
        # This ensures students with fewer preferences get MUCH higher priority
        if num_prefs > 0:
            priority_multiplier = 10 ** (max_total_prefs - num_prefs)
        else:
            priority_multiplier = 1
        
        # Assign weights: first preference gets highest weight
        # Use exponential weights based on preference rank (independent of total prefs)
        # Then multiply by priority_multiplier to boost students with fewer preferences
        for prof, pref_index in valid_prefs:
            # Rank is pref_index + 1 (1-based ranking)
            rank = pref_index + 1
            # Base exponential weighting: preference 1 gets 10^6, preference 2 gets 10^5, etc.
            # Use a fixed large base (max_total_prefs + 1) so all students use same scale
            base_weight = 10 ** (max_total_prefs + 1 - rank)
            # Apply priority multiplier to boost students with fewer preferences
            weight = base_weight * priority_multiplier
            student_prefs[prof] = weight
        
        preferences[student] = student_prefs

    # Define the optimization problem
    model = LpProblem("Meeting_Optimization", LpMaximize)

    # Create decision variables (student, professor, time slot)
    x = {
        (s, p, t): LpVariable(f"x_{s}_{p}_{t}", cat="Binary")
        for s in students for p in preferences.get(s, {}) for t in time_slots
    }

    # Objective Function: 
    # 1. Maximize weighted preference sum (primary objective)
    # 2. Minimize number of students per professor per slot (secondary objective)
    # 3. Prefer earlier time slots (tertiary objective)
    
    # Add a small bonus for earlier slots: slot 1 gets highest bonus, slot 2 gets slightly less, etc.
    # This ensures that when preference weights are equal, earlier slots are preferred
    # The bonus is small enough (1/1000 of preference weight) to not override preference priorities
    slot_bonus = {t: (len(time_slots) - time_slots.index(t)) / 1000.0 for t in time_slots}
    
    # Calculate a large base weight for preferences to ensure they dominate
    # Use the maximum preference weight as a reference
    max_pref_weight = 0
    for s in students:
        for p in preferences.get(s, {}):
            max_pref_weight = max(max_pref_weight, preferences[s][p])
    
    # Find minimum non-zero preference weight
    min_pref_weight = max_pref_weight
    for s in students:
        for p in preferences.get(s, {}):
            if preferences[s][p] > 0:
                min_pref_weight = min(min_pref_weight, preferences[s][p])
    
    # Create auxiliary variables to track group sizes (number of students per professor per slot)
    # This allows us to penalize larger groups more strongly
    group_size = {}
    for p in professors:
        for t in time_slots:
            # Count how many students are assigned to this professor in this slot
            group_size[p, t] = lpSum(x[s, p, t] for s in students if p in preferences.get(s, {}))
    
    # Penalty for group sizes: use a stronger penalty that increases with group size
    # Penalty = base_penalty * (group_size^2) to strongly discourage larger groups
    # But we need to linearize this, so we'll use a step penalty instead
    
    # Create binary variables to track if a professor has 2+ students, 3+ students, etc.
    # This allows us to add increasing penalties for larger groups
    has_multiple = {}
    for p in professors:
        for t in time_slots:
            # Binary variable: 1 if professor p has 2+ students in slot t
            has_multiple[p, t] = LpVariable(f"has_multiple_{p}_{t}", cat="Binary")
            # Constraint: has_multiple[p,t] = 1 if group_size[p,t] >= 2
            # group_size[p,t] >= 2 * has_multiple[p,t]
            model += group_size[p, t] >= 2 * has_multiple[p, t]
            # group_size[p,t] <= 1 + max_students * has_multiple[p,t]
            model += group_size[p, t] <= 1 + max_students * has_multiple[p, t]
    
    # Strong penalty for having multiple students (2+) in a group
    # This penalty should be significant but not override preference fulfillment
    multiple_student_penalty = min_pref_weight / 100.0  # Stronger penalty
    
    # Primary objective: maximize preference weights
    preference_objective = lpSum(
        (preferences[s][p] + slot_bonus[t]) * x[s, p, t] 
        for s in students for p in preferences.get(s, {}) for t in time_slots
    )
    
    # Secondary objective: minimize groups with multiple students
    # Penalize each professor-slot pair that has 2+ students
    group_penalty_term = lpSum(
        multiple_student_penalty * has_multiple[p, t]
        for p in professors for t in time_slots
    )
    
    # Combined objective: maximize preferences, minimize multiple-student groups
    model += preference_objective - group_penalty_term

    # Constraints

    # Each student can meet with up to "max_meetings" professors total (one per time slot)
    for s in students:
        model += lpSum(x[s, p, t] for p in preferences.get(s, {}) for t in time_slots) <= max_meetings

    # Each professor can meet with up to "max_meetings * max_students" students total
    for p in professors:
        model += lpSum(x[s, p, t] for s in students if p in preferences.get(s, {}) for t in time_slots) <= max_meetings * max_students

    # Each student can only have one meeting per time slot
    for s in students:
        for t in time_slots:
            model += lpSum(x[s, p, t] for p in preferences.get(s, {})) <= 1

    # Each professor can meet at most "max_students" in each time slot
    for p in professors:
        for t in time_slots:
            model += lpSum(x[s, p, t] for s in students if p in preferences.get(s, {})) <= max_students

    # Each student can meet each professor at most once
    for s in students:
        for p in preferences.get(s, {}):
            model += lpSum(x[s, p, t] for t in time_slots) <= 1

    # Professors can only be assigned in their available time slots
    for p in professors:
        for t in time_slots:
            if professor_df.loc[p, t] == 0:  # If professor is unavailable in this time slot
                for s in students:
                    if (s, p, t) in x:  # Only add constraint if variable exists
                        model += x[s, p, t] == 0

    # Solve the problem
    from pulp import LpStatus, getSolver
    # Try to use a better solver if available (CBC is usually better than default)
    try:
        solver = getSolver('PULP_CBC_CMD', msg=0)
        status = model.solve(solver)
    except:
        # Fall back to default solver
        status = model.solve()
    
    # Check if the solver found a solution
    status_str = LpStatus[status]
    if status_str != 'Optimal' and status_str != 'Feasible':
        # Provide more detailed error information
        error_msg = f"Optimization failed with status: {status_str}.\n"
        error_msg += "Possible reasons:\n"
        error_msg += "1. Not enough professors available for all student preferences\n"
        error_msg += "2. Professor availability conflicts with student preferences\n"
        error_msg += "3. Too many students per professor per slot (try reducing 'Students per Professor per Slot')\n"
        error_msg += "4. Students have preferences that cannot be satisfied given constraints\n"
        raise ValueError(error_msg)
    
    # Warn if solution is only feasible, not optimal
    if status_str == 'Feasible':
        import warnings
        warnings.warn("Solver found a feasible solution but may not be optimal. Consider checking constraints.")

    # Extract and display results in the requested format
    schedule = [(s, p, t) for (s, p, t) in x if x[s, p, t].varValue == 1]

    # Convert the schedule into a structured DataFrame where key = Student, value = meetings in order
    schedule_dict = {student: [None] * max_meetings for student in students}

    for student, professor, time_slot in schedule:
        time_index = time_slots.index(time_slot)
        schedule_dict[student][time_index] = professor

    # Convert to DataFrame
    schedule_df = pd.DataFrame.from_dict(schedule_dict, orient="index", columns=[f"Time Slot {i+1}" for i in range(max_meetings)])

    # Initialize dictionary to store professor schedules
    professor_schedule_dict = {professor: ["" for _ in range(max_meetings)] for professor in professors}

    # Populate the dictionary based on the optimized schedule
    for student, professor, time_slot in schedule:
        time_index = time_slots.index(time_slot)

        # If there is already a student assigned, append the new student
        if professor_schedule_dict[professor][time_index]:
            professor_schedule_dict[professor][time_index] += f"; {student}"
        else:
            professor_schedule_dict[professor][time_index] = student

    # Convert to DataFrame with professors as the index and time slots as columns
    professor_schedule_df = pd.DataFrame.from_dict(professor_schedule_dict, orient="index",
                                                   columns=[f"Time Slot {i+1}" for i in range(max_meetings)])

    # At the end of the function, add:
    preference_analysis = analyze_preference_fulfillment(schedule_df, students_df)
    
    return schedule_df, professor_schedule_df, preference_analysis

def analyze_preference_fulfillment(schedule_df, students_df):
    """
    Analyzes how many preferences each student got scheduled with.
    
    Args:
        schedule_df: DataFrame containing the optimized schedule
        students_df: DataFrame containing student preferences
    
    Returns:
        DataFrame with columns: Student, Preferences Met, Total Preferences, Ratio
    """
    results = []
    
    for student in schedule_df.index:
        # Get all professors this student is scheduled with (excluding None/empty slots)
        scheduled_profs = set(prof for prof in schedule_df.loc[student] if pd.notna(prof) and prof != '')
        
        # Get all preferences this student listed (excluding None/empty slots)
        preferred_profs = set(prof for prof in students_df.loc[student] if pd.notna(prof) and prof != ' ')
        
        # Calculate metrics
        preferences_met = len(scheduled_profs.intersection(preferred_profs))
        total_preferences = len(preferred_profs)
        ratio = f"{preferences_met}/{total_preferences}"
        
        results.append({
            'Student': student,
            'Preferences Met': preferences_met,
            'Total Preferences': total_preferences,
            'Ratio': ratio
        })
    
    return pd.DataFrame(results).set_index('Student')



