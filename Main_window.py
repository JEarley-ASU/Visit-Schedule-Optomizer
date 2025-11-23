from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
                               QMessageBox, QHeaderView, QTabWidget, QWidget, QTableWidget, QLineEdit, QTableWidgetItem,
                               QCheckBox, QFileDialog, QSpinBox)

from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

import sys
import json
import pandas as pd
import numpy as np
import s_optomize as sop

class ProfessorDialog(QDialog):
    def __init__(self, num_slots, header_spots):
        super().__init__()
        self.setWindowTitle("Add Professor")
        self.professor_last_name = QLineEdit()
        self.professor_first_name = QLineEdit()
        self.num_slots = num_slots
        self.checkboxes = []
        time_slot_names = header_spots[1:]

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Professor First Name:"))
        layout.addWidget(self.professor_first_name)

        layout.addWidget(QLabel("Professor Last Name:"))
        layout.addWidget(self.professor_last_name)

        layout.addWidget(QLabel("Unavailable Time Slots:"))

        for i in range(num_slots):

            checkbox = QCheckBox(f"{time_slot_names[i]}")
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        button_layout.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_data(self):
        prof_first_name = self.professor_first_name.text().strip().upper()
        prof_last_name = self.professor_last_name.text().strip().upper()
        name = prof_last_name + ", " + prof_first_name
        unavailable_slots = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
        return name, unavailable_slots

class MeetingScheduler(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Scheduler")
        self.showMaximized()
        self.num_slots = 6
        self.professor_count = 0
        self.saved_profs = 0
        self.visitor_count = 0
        self.name_memory = []
        self.professor_data = pd.DataFrame(columns=[f"Slot {i+1}" for i in range(self.num_slots)])
        self.visitor_data = pd.DataFrame(columns=[f"Preference {i+1}" for i in range(self.num_slots+3)])
        self.filename = None
        
        # Add this line to ensure the window stays open
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()
        self.init_tab = QWidget()
        self.create_tab = QWidget()
        self.visitor_tab = QWidget()
        self.schedule_tab = QWidget()  # Combined tab for both schedules

        self.tabs.addTab(self.init_tab, "Initialize")
        self.tabs.addTab(self.create_tab, "Faculty Info")
        self.tabs.addTab(self.visitor_tab, "Visitors Info")
        self.tabs.addTab(self.schedule_tab, "Schedules")  # Single tab for both schedules

        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)

        self.init_layout()
        self.create_layout()
        self.visitor_layout()
        self.schedule_layout()  # Updated layout for combined schedules

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def init_layout(self):
        layout = QVBoxLayout()

        # Add save/load buttons at the top
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_state)
        button_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_state)
        button_layout.addWidget(self.load_button)
        
        layout.addLayout(button_layout)

        layout.addWidget(QLabel("Number of Time Slots:"))
        self.slot_dropdown = QComboBox()
        self.slot_dropdown.addItems([str(i) for i in range(1, 11)])
        layout.addWidget(self.slot_dropdown)

        # Change init_button to be a class attribute so we can modify it
        self.init_button = QPushButton("Initialize")
        self.init_button.clicked.connect(self.handle_initialization)
        layout.addWidget(self.init_button)

        self.init_tab.setLayout(layout)

    def handle_initialization(self):
        """Handles both initial initialization and subsequent updates."""
        if self.init_button.text() == "Initialize":
            # First time initialization
            self.validate_initialization()
            self.init_button.setText("Update")  # Change button text
        else:
            # Handle updates to existing schedule
            self.update_schedule()

    def update_schedule(self):
        """Handles updating existing schedule with new number of slots."""
        old_num_slots = self.num_slots
        new_num_slots = int(self.slot_dropdown.currentText())
        
        if old_num_slots != new_num_slots:
            # Update the tables' structure
            self.num_slots = new_num_slots
            self.update_schedule_table()
            self.update_vistor_table()
            
            # Adjust existing data and refresh UI
            self.adjust_data_for_new_slots(old_num_slots, new_num_slots)
            
            QMessageBox.information(self, "Success", f"Schedule updated to {new_num_slots} time slots!")
        else:
            QMessageBox.information(self, "No Change", "Number of time slots remains the same.")

    def create_layout(self):
        layout = QVBoxLayout()

        # Add professor name save/load buttons at the top
        button_layout = QHBoxLayout()
        
        self.save_profs_button = QPushButton("Save Professors")
        self.save_profs_button.clicked.connect(self.save_professor_names)
        button_layout.addWidget(self.save_profs_button)
        
        self.load_profs_button = QPushButton("Load Professors")
        self.load_profs_button.clicked.connect(self.load_professor_names)
        button_layout.addWidget(self.load_profs_button)
        
        layout.addLayout(button_layout)

        # Professor table
        self.professor_table = QTableWidget()
        self.professor_table.setColumnCount(self.num_slots + 1)
        self.professor_table.setHorizontalHeaderLabels(["Professor"] + [f"Slot {i + 1}" for i in range(self.num_slots)])
        self.professor_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.professor_table.itemChanged.connect(self.update_professor_name)
        layout.addWidget(self.professor_table)

        # Add/Remove professor buttons
        button_layout2 = QHBoxLayout()
        self.add_professor_button = QPushButton("+")
        self.add_professor_button.setFixedSize(40, 40)
        self.add_professor_button.clicked.connect(self.open_professor_dialog)
        button_layout2.addWidget(self.add_professor_button)

        self.remove_professor_button = QPushButton("-")
        self.remove_professor_button.setFixedSize(40, 40)
        self.remove_professor_button.clicked.connect(self.remove_selected_professor)
        button_layout2.addWidget(self.remove_professor_button)
        
        layout.addLayout(button_layout2)
        self.create_tab.setLayout(layout)

    def visitor_layout(self):
        layout = QVBoxLayout()

        self.visitor_table = QTableWidget()
        self.visitor_table.setColumnCount(self.num_slots + 4)
        self.visitor_table.setHorizontalHeaderLabels(["Visitor"] + [f"Preference {i + 1}" for i in range(self.num_slots + 3)])
        self.visitor_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.visitor_table)

        button_layout = QHBoxLayout()
        self.add_visitor_button = QPushButton("+")
        self.add_visitor_button.setFixedSize(40, 40)
        self.add_visitor_button.clicked.connect(self.add_blank_visitor_row)
        button_layout.addWidget(self.add_visitor_button)

        self.remove_visitor_button = QPushButton("-")
        self.remove_visitor_button.setFixedSize(40, 40)
        self.remove_visitor_button.clicked.connect(self.remove_selected_visitor)
        button_layout.addWidget(self.remove_visitor_button)
        
        layout.addLayout(button_layout)
        self.visitor_tab.setLayout(layout)

    def add_blank_visitor_row(self):
        """Adds a new visitor row with a default placeholder name and dropdowns for professor preferences."""
        row_count = self.visitor_table.rowCount()
        self.visitor_table.insertRow(row_count)

        # Default visitor name
        default_name = f"NEW ENTRY {row_count + 1}"
        self.name_memory.append(default_name)  # Track initial name

        # Set first column as an editable field for visitor name
        item = QTableWidgetItem(default_name)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.visitor_table.setItem(row_count, 0, item)

        # Connect name change event
        self.visitor_table.itemChanged.connect(self.update_visitor_name)

        # Populate preference columns with dropdowns
        professor_names = sorted(list(self.professor_data.index.to_list()))
        for col in range(1, self.num_slots + 4):
            combo = QComboBox()
            combo.addItem(" ")  # Default empty selection
            combo.addItems(professor_names)
            combo.currentIndexChanged.connect(
                lambda _, r=row_count, c=col, cb=combo: self.update_visitor_data(r, c, cb))
            self.visitor_table.setCellWidget(row_count, col, combo)

        # Initialize visitor row in the DataFrame
        self.visitor_data.loc[default_name] = [" "] * self.visitor_data.shape[1]

    def update_visitor_name(self, item):
        """Updates the visitor's name dynamically when changed in the table."""
        row = item.row()
        new_name = item.text().strip()

        # Get the old name from memory
        old_name = self.name_memory[row] if row < len(self.name_memory) else f"Visitor {row + 1}"

        if new_name and new_name != old_name:
            if old_name in self.visitor_data.index:
                self.visitor_data.rename(index={old_name: new_name}, inplace=True)
            else:
                print(f"Warning: {old_name} not found in visitor_data. Adding new entry.")
                self.visitor_data.loc[new_name] = self.visitor_data.loc[
                    old_name] if old_name in self.visitor_data.index else [" "] * self.visitor_data.shape[1]

            # Update the name in memory
            self.name_memory[row] = new_name

            # Update the UI to reflect the new name
            self.visitor_table.item(row, 0).setText(new_name)

    def schedule_layout(self):
        """Creates a stacked layout for the professor and visitor schedules with controls below."""
        main_layout = QVBoxLayout()

        # Tables layout (horizontally stacked)
        tables_layout = QHBoxLayout()

        # Visitor Schedule Table
        self.visitor_sched_table = QTableWidget()
        self.visitor_sched_table.setColumnCount(self.num_slots + 1)
        self.visitor_sched_table.setHorizontalHeaderLabels(
            ["Visitor"] + [f"Meeting {i + 1}" for i in range(self.num_slots)])
        self.visitor_sched_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tables_layout.addWidget(self.visitor_sched_table)

        # Professor Schedule Table
        self.professor_sched_table = QTableWidget()
        self.professor_sched_table.setColumnCount(self.num_slots + 1)
        self.professor_sched_table.setHorizontalHeaderLabels(
            ["Professor"] + [f"Meeting {i + 1}" for i in range(self.num_slots)])
        self.professor_sched_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tables_layout.addWidget(self.professor_sched_table)

        main_layout.addLayout(tables_layout)

        # Controls layout (stacked below tables)
        controls_layout = QVBoxLayout()

        # Student density control
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Students per Professor per Slot:"))
        self.student_density_spinner = QSpinBox()
        self.student_density_spinner.setRange(1, 5)
        self.student_density_spinner.setValue(1)
        density_layout.addWidget(self.student_density_spinner)
        controls_layout.addLayout(density_layout)

        # Buttons
        self.optimize_button = QPushButton("Optimize Schedule")
        self.optimize_button.clicked.connect(self.optimize)
        controls_layout.addWidget(self.optimize_button)

        self.save_sched_as_button = QPushButton("Export As")
        self.save_sched_as_button.clicked.connect(self.save_as_schedules_to_excel)
        controls_layout.addWidget(self.save_sched_as_button)

        self.save_sched_button = QPushButton("Export")
        self.save_sched_button.clicked.connect(self.save_schedules_to_excel)
        controls_layout.addWidget(self.save_sched_button)

        main_layout.addLayout(controls_layout)
        self.schedule_tab.setLayout(main_layout)

    def update_visitor_data(self, row, col, combo):
        """Updates the visitor_data DataFrame when a dropdown changes and dynamically updates visitor name."""
        visitor_name_item = self.visitor_table.item(row, 0)
        if visitor_name_item:
            visitor_name = visitor_name_item.text().strip()
            if visitor_name:
                self.visitor_data.at[visitor_name, f"Preference {col}"] = combo.currentText()

    def visitor_sched_layout(self):
        layout = QVBoxLayout()

        self.visitor_sched_table = QTableWidget()
        self.visitor_sched_table.setColumnCount(self.num_slots + 1)
        self.visitor_sched_table.setHorizontalHeaderLabels(["Visitor"] + [f"Slot {i + 1}" for i in range(self.num_slots)])
        self.visitor_sched_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.visitor_sched_table)
        self.Visitor_Sched_tab.setLayout(layout)


    def update_schedule_table(self):
        self.num_slots = int(self.slot_dropdown.currentText())

        self.headers = ["Professor"]
        for i in range(self.num_slots):
            self.headers.append(f"Meeting Time {i+1}")

        # Update professor table columns
        self.professor_table.setColumnCount(self.num_slots + 1)
        self.professor_table.setHorizontalHeaderLabels(self.headers)

        # Update schedule tables columns
        self.visitor_sched_table.setColumnCount(self.num_slots + 1)
        self.visitor_sched_table.setHorizontalHeaderLabels(["Visitor"] + [f"Meeting {i + 1}" for i in range(self.num_slots)])
        
        self.professor_sched_table.setColumnCount(self.num_slots + 1)
        self.professor_sched_table.setHorizontalHeaderLabels(["Professor"] + [f"Meeting {i + 1}" for i in range(self.num_slots)])

    def update_vistor_table(self):
        self.num_slots = int(self.slot_dropdown.currentText())
        self.visitor_headers = ["Visitor"]
        for i in range(self.num_slots+3):
            self.visitor_headers.append(f"Preference #{i+1}")
        
        # Update visitor table columns
        self.visitor_table.setColumnCount(self.num_slots + 4)  # +4 because we have Visitor column and 3 extra preference slots
        self.visitor_table.setHorizontalHeaderLabels(self.visitor_headers)

    def open_professor_dialog(self):
        """Opens the ProfessorDialog and updates professor_data."""
        dialog = ProfessorDialog(self.num_slots, self.headers)

        if dialog.exec():
            name, unavailable_slots = dialog.get_data()
            if name:
                self.add_professor_to_table(name, unavailable_slots)

                # Initialize professor availability
                availability = np.ones(self.num_slots, dtype=int)
                for slot in unavailable_slots:
                    availability[slot] = 0  # Mark unavailable

                # Ensure the DataFrame has the correct columns
                if self.professor_data.empty:
                    self.professor_data = pd.DataFrame(columns=[f"Slot {i + 1}" for i in range(self.num_slots)])

                # Update existing DataFrame dynamically
                self.professor_data.loc[name] = availability
                self.update_professor_dropdowns()  # Update visitor dropdowns
            else:
                QMessageBox.warning(self, "Warning", "Professor name cannot be empty.")

    def add_professor_to_table(self, name, unavailable_slots):
        """Adds a new professor row with dropdowns for availability."""
        row = self.professor_count
        self.professor_table.insertRow(row)
        
        # Make professor name editable
        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
        self.professor_table.setItem(row, 0, name_item)

        for col in range(1, self.num_slots + 1):
            combo = QComboBox()
            combo.addItems(["Available", "Unavailable"])  # Dropdown options
            combo.setCurrentText("Unavailable" if (col - 1) in unavailable_slots else "Available")

            # Apply background color
            self.style_availability_dropdown(combo)

            # Connect change signal to dynamically update DataFrame
            combo.currentIndexChanged.connect(lambda _, c=combo, n=name, s=col - 1: self.update_professor_data(n, s, c))

            self.professor_table.setCellWidget(row, col, combo)

        self.professor_count += 1

    def update_professor_name(self, item):
        """Updates the professor's name dynamically when changed in the table."""
        if item.column() == 0:  # Only process changes to the name column
            row = item.row()
            new_name = item.text().strip().upper()
            
            # Get the old name from the professor_data index
            if row < len(self.professor_data.index):
                old_name = self.professor_data.index[row]
                
                if new_name and new_name != old_name:
                    try:
                        # Update professor_data
                        self.professor_data.rename(index={old_name: new_name}, inplace=True)

                        # Update visitor preferences with the new name
                        for row_idx in range(self.visitor_table.rowCount()):
                            for col_idx in range(1, self.visitor_table.columnCount()):
                                combo = self.visitor_table.cellWidget(row_idx, col_idx)
                                if combo and combo.currentText() == old_name:
                                    combo.setCurrentText(new_name)

                        # Update visitor_data with the new name
                        for col in self.visitor_data.columns:
                            self.visitor_data[col] = self.visitor_data[col].replace(old_name, new_name)

                        # Update all visitor preference dropdowns with the new name list
                        self.update_professor_dropdowns()

                        # Update the UI to reflect the new name
                        self.professor_table.item(row, 0).setText(new_name)

                        # Update the combo box connections for this row
                        for col in range(1, self.num_slots + 1):
                            combo = self.professor_table.cellWidget(row, col)
                            if combo:
                                # Disconnect old connections (if any) and connect new one
                                try:
                                    combo.currentIndexChanged.disconnect()
                                except TypeError:
                                    pass
                                combo.currentIndexChanged.connect(
                                    lambda _, c=combo, n=new_name, s=col-1: 
                                    self.update_professor_data(n, s, c))
                                    
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to update professor name: {str(e)}")
                        # Revert the name in the table
                        self.professor_table.item(row, 0).setText(old_name)

    def update_professor_data(self, professor_name, slot, combo):
        """Updates the professor_data DataFrame when a dropdown changes."""
        self.professor_data.at[professor_name, f"Slot {slot + 1}"] = 1 if combo.currentText() == "Available" else 0
        self.style_availability_dropdown(combo)

    def update_professor_dropdowns(self):
        """Updates all dropdowns in the visitor table with the latest professor list."""
        professor_names = sorted(list(self.professor_data.index))  # Ensure alphabetical order

        for row in range(self.visitor_table.rowCount()):
            for col in range(1, self.num_slots + 4):  # Skip first column (visitor name)
                combo = self.visitor_table.cellWidget(row, col)
                if isinstance(combo, QComboBox):
                    current_selection = combo.currentText()  # Preserve selection if possible
                    combo.clear()
                    combo.addItem(" ")  # Default empty option
                    combo.addItems(professor_names)

                    # Restore the previous selection if it exists
                    if current_selection in professor_names:
                        combo.setCurrentText(current_selection)


    def style_availability_dropdown(self, combo):
        """Applies background color to the dropdown based on selected value."""
        if combo.currentText() == "Available":
            combo.setStyleSheet("QComboBox { background-color: lightgreen; }")
        else:
            combo.setStyleSheet("QComboBox { background-color: lightcoral; }")

    def optimize(self):
        time_slots = [f"Slot {i+1}" for i in range(self.num_slots)]
        max_students = self.student_density_spinner.value()
        
        try:
            visitor_schedule, professor_schedule, preference_analysis = sop.generate_schedule(
                self.visitor_data, 
                self.professor_data, 
                time_slots,
                max_students=max_students
            )
            
            # Store schedules as class attributes for later use
            self.visitor_schedule = visitor_schedule
            self.professor_schedule = professor_schedule
            
            # Display visitor schedule
            self.visitor_sched_table.setRowCount(len(visitor_schedule))
            for i, (visitor, row) in enumerate(visitor_schedule.iterrows()):
                self.visitor_sched_table.setItem(i, 0, QTableWidgetItem(visitor))
                for j, prof in enumerate(row):
                    self.visitor_sched_table.setItem(i, j + 1, 
                        QTableWidgetItem(str(prof) if pd.notna(prof) else ""))
            
            # Display professor schedule
            self.professor_sched_table.setRowCount(len(professor_schedule))
            for i, (professor, row) in enumerate(professor_schedule.iterrows()):
                self.professor_sched_table.setItem(i, 0, QTableWidgetItem(professor))
                for j, visitors in enumerate(row):
                    self.professor_sched_table.setItem(i, j + 1, 
                        QTableWidgetItem(str(visitors) if pd.notna(visitors) else ""))
            
            # Color code visitor preferences based on schedule
            self.color_code_preferences(visitor_schedule)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Optimization failed: {str(e)}")

    def color_code_preferences(self, schedule):
        """Colors visitor preference cells based on whether they got their preferences."""
        for row in range(self.visitor_table.rowCount()):
            visitor_name = self.visitor_table.item(row, 0).text()
            
            # Get all professors this visitor is scheduled with
            scheduled_profs = set(prof for prof in schedule.loc[visitor_name] if pd.notna(prof) and prof != '')
            
            # Check each preference cell
            for col in range(1, self.visitor_table.columnCount()):
                combo = self.visitor_table.cellWidget(row, col)
                if combo and isinstance(combo, QComboBox):
                    prof_name = combo.currentText()
                    if prof_name and prof_name != " ":
                        if prof_name in scheduled_profs:
                            combo.setStyleSheet("QComboBox { background-color: lightgreen; }")
                        else:
                            combo.setStyleSheet("QComboBox { background-color: lightcoral; }")
                    else:
                        combo.setStyleSheet("")  # Reset style for empty preferences

    def add_visitor_to_table(self, name, preferences):
        row = self.visitor_count
        self.visitor_table.insertRow(row)
        self.visitor_table.setItem(row, 0, QTableWidgetItem(name))

        for col in range(1, self.num_slots + 3):

            self.visitor_table.setItem(row, col, QTableWidgetItem(preferences[col-1]))

        self.visitor_count += 1

    def validate_initialization(self):
        old_num_slots = self.num_slots
        new_num_slots = int(self.slot_dropdown.currentText())
        
        # If we have existing data, adjust it for the new number of slots
        if not self.professor_data.empty or not self.visitor_data.empty:
            self.adjust_data_for_new_slots(old_num_slots, new_num_slots)
        
        self.num_slots = new_num_slots
        self.update_schedule_table()
        self.update_vistor_table()
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        self.tabs.setTabEnabled(3, True)
        QMessageBox.information(self, "Confirmation", f"Schedule initialized successfully!")
        self.tabs.setCurrentWidget(self.create_tab)

    def adjust_data_for_new_slots(self, old_slots, new_slots):
        """Adjusts existing data when changing the number of time slots."""
        # Adjust professor data
        if not self.professor_data.empty:
            if new_slots > old_slots:
                # Add new columns with default availability (1 for available)
                for i in range(old_slots, new_slots):
                    self.professor_data[f"Slot {i + 1}"] = 1
            else:
                # Remove excess columns
                for i in range(new_slots, old_slots):
                    del self.professor_data[f"Slot {i + 1}"]

        # Adjust visitor data
        if not self.visitor_data.empty:
            old_pref_cols = old_slots + 3
            new_pref_cols = new_slots + 3
            
            if new_pref_cols > old_pref_cols:
                # Add new columns with empty preferences
                for i in range(old_pref_cols, new_pref_cols):
                    self.visitor_data[f"Preference {i + 1}"] = " "
            else:
                # Remove excess columns
                for i in range(new_pref_cols, old_pref_cols):
                    del self.visitor_data[f"Preference {i + 1}"]

        # Update tables to reflect new data
        if not self.professor_data.empty:
            self.professor_table.setRowCount(0)  # Clear existing rows
            for professor_name, row_data in self.professor_data.iterrows():
                row = self.professor_table.rowCount()
                self.professor_table.insertRow(row)
                self.professor_table.setItem(row, 0, QTableWidgetItem(professor_name))
                
                # Add dropdowns for all columns
                for col in range(self.num_slots):
                    combo = QComboBox()
                    combo.addItems(["Available", "Unavailable"])
                    # Set to "Available" for new columns, otherwise use existing data
                    value = row_data[f"Slot {col + 1}"] if col < old_slots else 1
                    combo.setCurrentText("Available" if value == 1 else "Unavailable")
                    self.style_availability_dropdown(combo)
                    combo.currentIndexChanged.connect(
                        lambda state, name=professor_name, slot=col, cb=combo: 
                        self.update_professor_data(name, slot, cb))
                    self.professor_table.setCellWidget(row, col + 1, combo)

        if not self.visitor_data.empty:
            self.visitor_table.setRowCount(0)  # Clear existing rows
            professor_names = sorted(list(self.professor_data.index))
            
            for visitor_name, row_data in self.visitor_data.iterrows():
                row = self.visitor_table.rowCount()
                self.visitor_table.insertRow(row)
                name_item = QTableWidgetItem(visitor_name)
                name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
                self.visitor_table.setItem(row, 0, name_item)
                
                # Add dropdowns for all columns
                for col in range(self.num_slots + 3):
                    combo = QComboBox()
                    combo.addItem(" ")  # Empty preference
                    combo.addItems(professor_names)
                    # Use existing value for old columns, empty for new ones
                    value = str(row_data[f"Preference {col + 1}"]) if col < old_pref_cols else " "
                    combo.setCurrentText(value)
                    combo.currentIndexChanged.connect(
                        lambda state, r=row, c=col+1, cb=combo: 
                        self.update_visitor_data(r, c, cb))
                    self.visitor_table.setCellWidget(row, col + 1, combo)

    def update_schedule_tables(self):
        """Updates both visitor and professor schedule tables."""
        # Update Visitor Schedule Table
        if self.visitor_schedule is not None and not self.visitor_schedule.empty:
            self.visitor_sched_table.setRowCount(self.visitor_schedule.shape[0])
            self.visitor_sched_table.setColumnCount(self.visitor_schedule.shape[1])
            self.visitor_sched_table.setHorizontalHeaderLabels(self.visitor_schedule.columns.astype(str))
            self.visitor_sched_table.setVerticalHeaderLabels(self.visitor_schedule.index.astype(str))

            for row in range(self.visitor_schedule.shape[0]):
                for col in range(self.visitor_schedule.shape[1]):
                    value = str(self.visitor_schedule.iloc[row, col]) if pd.notna(self.visitor_schedule.iloc[row, col]) else " "
                    self.visitor_sched_table.setItem(row, col, QTableWidgetItem(value))

        # Update Professor Schedule Table
        if self.professor_schedule is not None and not self.professor_schedule.empty:
            self.professor_sched_table.setRowCount(self.professor_schedule.shape[0])
            self.professor_sched_table.setColumnCount(self.professor_schedule.shape[1])
            self.professor_sched_table.setHorizontalHeaderLabels(self.professor_schedule.columns.astype(str))
            self.professor_sched_table.setVerticalHeaderLabels(self.professor_schedule.index.astype(str))

            for row in range(self.professor_schedule.shape[0]):
                for col in range(self.professor_schedule.shape[1]):
                    value = str(self.professor_schedule.iloc[row, col]) if pd.notna(self.professor_schedule.iloc[row, col]) else " "
                    self.professor_sched_table.setItem(row, col, QTableWidgetItem(value))


    def save_as_schedules_to_excel(self):
        """Opens a file dialog and saves both professor_data and visitor_data as separate sheets in an Excel file."""
        self.filename, _ = QFileDialog.getSaveFileName(self, "Export Schedule", "", "Excel Files (*.xlsx);;All Files (*)")
        if self.filename:
            with pd.ExcelWriter(self.filename) as writer:
                self.professor_schedule.to_excel(writer, sheet_name="Professor Schedule")
                self.visitor_schedule.to_excel(writer, sheet_name="Visitor Schedule")
            QMessageBox.information(self, "Success", f"Schedules exported to {self.filename}")

    def save_schedules_to_excel(self):
        """Opens a file dialog and saves both professor_data and visitor_data as separate sheets in an Excel file."""
        if self.filename:
            with pd.ExcelWriter(self.filename) as writer:
                self.professor_schedule.to_excel(writer, sheet_name="Professor Schedule")
                self.visitor_schedule.to_excel(writer, sheet_name="Visitor Schedule")
            QMessageBox.information(self, "Success", f"Schedules saved to {self.filename}")
        else:
            self.filename, _ = QFileDialog.getSaveFileName(self, "Export Schedule", "", "Excel Files (*.xlsx);;All Files (*)")
            with pd.ExcelWriter(self.filename) as writer:
                self.professor_schedule.to_excel(writer, sheet_name="Professor Schedule")
                self.visitor_schedule.to_excel(writer, sheet_name="Visitor Schedule")
            QMessageBox.information(self, "Success", f"Schedules exported to {self.filename}")

    def save_state(self):
        """Saves the current state of the program to a JSON file."""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Progress", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            state = {
                'num_slots': self.num_slots,
                'professor_count': self.professor_count,
                'visitor_count': self.visitor_count,
                'name_memory': self.name_memory,
                'professor_data': self.professor_data.to_dict(),
                'visitor_data': self.visitor_data.to_dict(),
                'current_tab': self.tabs.currentIndex(),
                'slot_dropdown_value': self.slot_dropdown.currentText()
            }
            
            try:
                with open(filename, 'w') as f:
                    json.dump(state, f)
                QMessageBox.information(self, "Success", "Progress saved successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save progress: {str(e)}")

    def load_state(self):
        """Loads a previously saved state from a JSON file."""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Progress", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    state = json.load(f)
                
                # Restore program state
                self.num_slots = state['num_slots']
                self.professor_count = state['professor_count']
                self.visitor_count = state['visitor_count']
                self.name_memory = state['name_memory']
                
                # Restore DataFrames
                self.professor_data = pd.DataFrame.from_dict(state['professor_data'])
                self.visitor_data = pd.DataFrame.from_dict(state['visitor_data'])
                
                # Update UI
                self.slot_dropdown.setCurrentText(state['slot_dropdown_value'])
                self.init_button.setText("Update")  # Change button text after loading
                self.validate_initialization()
                
                # Restore tables
                self.restore_professor_table()
                self.restore_visitor_table()
                
                # Enable tabs and set current tab
                self.tabs.setTabEnabled(1, True)
                self.tabs.setTabEnabled(2, True)
                self.tabs.setTabEnabled(3, True)
                self.tabs.setCurrentIndex(state['current_tab'])
                
                QMessageBox.information(self, "Success", "Progress loaded successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load progress: {str(e)}")

    def restore_professor_table(self):
        """Restores the professor table from the loaded professor_data."""
        self.professor_table.setRowCount(0)
        for professor_name, row_data in self.professor_data.iterrows():
            row = self.professor_table.rowCount()
            self.professor_table.insertRow(row)
            self.professor_table.setItem(row, 0, QTableWidgetItem(professor_name))
            
            for col, value in enumerate(row_data):
                combo = QComboBox()
                combo.addItems(["Available", "Unavailable"])
                combo.setCurrentText("Available" if value == 1 else "Unavailable")
                self.style_availability_dropdown(combo)
                
                # Create a new lambda function for each combo box
                combo.currentIndexChanged.connect(
                    lambda state, name=professor_name, slot=col, cb=combo: 
                    self.update_professor_data(name, slot, cb))
                    
                self.professor_table.setCellWidget(row, col + 1, combo)

    def restore_visitor_table(self):
        """Restores the visitor table from the loaded visitor_data."""
        self.visitor_table.setRowCount(0)
        professor_names = sorted(list(self.professor_data.index))
        
        for visitor_name, row_data in self.visitor_data.iterrows():
            row = self.visitor_table.rowCount()
            self.visitor_table.insertRow(row)
            
            # Set visitor name and make it editable
            name_item = QTableWidgetItem(visitor_name)
            name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
            self.visitor_table.setItem(row, 0, name_item)
            
            # Add to name memory
            if visitor_name not in self.name_memory:
                self.name_memory.append(visitor_name)
            
            for col, value in enumerate(row_data):
                combo = QComboBox()
                combo.addItem(" ")
                combo.addItems(professor_names)
                combo.setCurrentText(str(value))
                
                # Connect the signal handler for preference updates
                combo.currentIndexChanged.connect(
                    lambda state, r=row, c=col+1, cb=combo: 
                    self.update_visitor_data(r, c, cb))
                    
                self.visitor_table.setCellWidget(row, col + 1, combo)
        
        # Reconnect the name change handler
        self.visitor_table.itemChanged.connect(self.update_visitor_name)

    def remove_selected_professor(self):
        """Removes the selected professor row and updates the data."""
        selected_items = self.professor_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a professor to remove.")
            return

        selected_row = selected_items[0].row()
        professor_name = self.professor_table.item(selected_row, 0).text()

        reply = QMessageBox.question(self, 'Confirm Deletion',
                                   f'Are you sure you want to remove {professor_name}?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Remove from DataFrame
            self.professor_data.drop(professor_name, inplace=True)
            
            # Remove from table
            self.professor_table.removeRow(selected_row)
            self.professor_count -= 1

            # Update visitor preferences (remove this professor from dropdowns)
            self.update_professor_dropdowns()

    def remove_selected_visitor(self):
        """Removes the selected visitor row and updates the data."""
        selected_items = self.visitor_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a visitor to remove.")
            return

        selected_row = selected_items[0].row()
        visitor_name = self.visitor_table.item(selected_row, 0).text()

        reply = QMessageBox.question(self, 'Confirm Deletion',
                                   f'Are you sure you want to remove {visitor_name}?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Remove from DataFrame
            self.visitor_data.drop(visitor_name, inplace=True)
            
            # Remove from name memory
            if visitor_name in self.name_memory:
                self.name_memory.remove(visitor_name)
            
            # Remove from table
            self.visitor_table.removeRow(selected_row)
            self.visitor_count -= 1

    def save_professor_names(self):
        """Saves just the professor names to a JSON file."""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Professor Names", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            try:
                professor_names = sorted(list(self.professor_data.index))
                with open(filename, 'w') as f:
                    json.dump({"professor_names": professor_names}, f)
                QMessageBox.information(self, "Success", "Professor names saved successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save professor names: {str(e)}")

    def load_professor_names(self):
        """Loads professor names and initializes them with current number of slots."""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Professor Names", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    professor_names = sorted(data["professor_names"])  # Sort alphabetically

                # Clear existing professor data
                self.professor_table.setRowCount(0)
                self.professor_data = pd.DataFrame(columns=[f"Slot {i + 1}" for i in range(self.num_slots)])
                self.professor_count = 0

                # Add each professor with default availability (all available)
                for name in professor_names:
                    self.add_professor_to_table(name, [])  # Empty list means no unavailable slots
                    self.professor_data.loc[name] = [1] * self.num_slots  # All slots available

                # Update visitor dropdowns with new professor list
                self.update_professor_dropdowns()
                
                QMessageBox.information(self, "Success", "Professor names loaded successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load professor names: {str(e)}")
