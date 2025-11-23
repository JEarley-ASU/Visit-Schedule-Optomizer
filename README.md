# Meeting Scheduler - Streamlit Version

A web-based application for scheduling meetings between visitors/students and professors using optimization algorithms.

## Features

- **Initialize**: Set the number of time slots for meetings
- **Faculty Info**: Add professors and manage their availability for each time slot
- **Visitors Info**: Add visitors and set their professor preferences
- **Schedules**: Optimize and view generated meeting schedules with preference fulfillment analysis
- **Save/Load**: Save and load your progress as JSON files
- **Export**: Export optimized schedules to Excel format

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy pulp openpyxl
```

## Running the Application

To run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Initialize Tab**:
   - Select the number of time slots (1-10)
   - Click "Initialize" to set up the schedule
   - Use "Save Progress" to download your current state
   - Use "Load Progress" to restore a previously saved state

2. **Faculty Info Tab**:
   - Add professors by entering their first and last names
   - Mark unavailable time slots for each professor
   - Edit professor availability directly in the table
   - Remove professors as needed
   - Save/load professor names separately

3. **Visitors Info Tab**:
   - Add visitors by entering their names
   - Set preferences for each visitor (up to number of slots + 3 preferences)
   - Edit visitor preferences in the editable table
   - Remove visitors as needed

4. **Schedules Tab**:
   - Set the maximum number of students per professor per slot
   - Click "Optimize Schedule" to generate the optimal meeting schedule
   - View visitor and professor schedules side by side
   - See preference fulfillment analysis
   - Export schedules to Excel format

## File Structure

- `app.py` - Main Streamlit application
- `s_optomize.py` - Optimization algorithm using linear programming
- `requirements.txt` - Python dependencies
- `main.py` - Original PySide6 desktop application (for reference)
- `Main_window.py` - Original PySide6 GUI code (for reference)

## Notes

- The optimization algorithm uses linear programming to maximize preference fulfillment
- Preferences are weighted (higher preference = higher weight)
- The algorithm ensures:
  - Each visitor meets with at most one professor per time slot
  - Each professor meets with at most the specified number of visitors per slot
  - Professors are only scheduled during their available time slots
  - Each visitor-professor pair meets at most once

