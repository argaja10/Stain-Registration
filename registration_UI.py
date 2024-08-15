import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QInputDialog, QDialog, QFormLayout, QLineEdit, QDialogButtonBox
from PyQt5.QtGui import QPixmap
import cv2
import subprocess

# Assuming these are the three main processing methods from the project
# import stain_registration  

class StainRegistrationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Stain Registration")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.target_button = QPushButton("Target Image", self)
        self.target_button.clicked.connect(self.target_image)
        self.layout.addWidget(self.target_button)

        self.source_button = QPushButton("Source Image", self)
        self.source_button.clicked.connect(self.source_image)
        self.layout.addWidget(self.source_button)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        # Buttons for the three methods
        self.method1_button = QPushButton("Method 1: SIFT", self)
        self.method1_button.clicked.connect(self.method1_parameters)
        self.layout.addWidget(self.method1_button)

        self.method2_button = QPushButton("Method 2: Optical Flow", self)
        self.method2_button.clicked.connect(self.method2_parameters)
        self.layout.addWidget(self.method2_button)

        self.method3_button = QPushButton("Method 3: ANTs", self)
        self.method3_button.clicked.connect(self.method3_parameters)
        self.layout.addWidget(self.method3_button)

        self.result_label = QLabel(self)
        self.layout.addWidget(self.result_label)

    def target_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.target_image= file_name
            #pixmap = QPixmap(self.image_path)
            #self.image_label.setPixmap(pixmap)
    
    def source_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.source_image= file_name

    def method1_parameters(self):
        dialog = Method1Dialog(self)
        if dialog.exec_():
            param1, param2, param3, param4 = dialog.getInputs()
            self.process_method1(param1, param2, param3, param4)

    def method2_parameters(self):
        dialog = Method2Dialog(self)
        if dialog.exec_():
            param1, param2, param3 = dialog.getInputs()
            self.process_method2(param1, param2, param3)

    def method3_parameters(self):
        dialog = Method2Dialog(self)
        if dialog.exec_():
            param1, param2, param3 = dialog.getInputs()
            self.process_method2(param1, param2, param3)

    def process_method1(self, param1, param2, param3, param4):

        try:
            command = [
            "python", "stain_registration/template_matching_SIFT1.py",
            "--source_path", self.source_image,
            "--normalized_target_path", param1,
            "--target_path", self.target_image,
            "--source_landmarks_path", param2,
            "--target_landmarks_path", param3,
            "--flag", param4
               ]
        
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)
        
            # Check if the command was successful
            if result.returncode == 0:
                print("Command executed successfully")
                print(result.stdout)  # Print the output from the script
            else:
                print("Error occurred")
                print(result.stderr)  # Print the error output from the script
        
        except Exception as e:
            print(f"An error occurred: {e}")

    def process_method2(self, param1, param2, param3):
        try:
            command = [
            "python", "stain_registration/optical_flow.py",
            "--source_path", self.source_image,
            "--normalized_target_path", param1,
            "--source_landmarks_path", param2,
            "--target_landmarks_path", param3
               ]
        
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)
        
            # Check if the command was successful
            if result.returncode == 0:
                print("Command executed successfully")
                print(result.stdout)  # Print the output from the script
            else:
                print("Error occurred")
                print(result.stderr)  # Print the error output from the script
        
        except Exception as e:
            print(f"An error occurred: {e}")


    def process_method3(self, param1, param2, param3):
        try:
            command = [
            "python", "stain_registration/ants_method.py",
            "--source_path", self.source_image,
            "--normalized_target_path", param1,
            "--source_landmarks_path", param2,
            "--target_landmarks_path", param3
               ]
        
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)
        
            # Check if the command was successful
            if result.returncode == 0:
                print("Command executed successfully")
                print(result.stdout)  # Print the output from the script
            else:
                print("Error occurred")
                print(result.stderr)  # Print the error output from the script
        
        except Exception as e:
            print(f"An error occurred: {e}")



class Method1Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Method 1 Parameters")

        self.layout = QFormLayout(self)

        self.param1_input = QLineEdit(self)
        self.param2_input = QLineEdit(self)
        self.param3_input = QLineEdit(self)
        self.param4_input = QLineEdit(self)
        
        self.layout.addRow("Normalized target path:", self.param1_input)
        self.layout.addRow("Source landmarks:", self.param2_input)
        self.layout.addRow("Target landmarks:", self.param3_input)
        self.layout.addRow("Flag:", self.param4_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.layout.addWidget(self.buttons)

    def getInputs(self):
        return self.param1_input.text(), self.param2_input.text(), self.param3_input.text(), self.param4_input.text()


class Method2Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Method 2 Parameters")

        self.layout = QFormLayout(self)

        self.param1_input = QLineEdit(self)
        self.param2_input = QLineEdit(self)
        self.param3_input = QLineEdit(self)
        
        self.layout.addRow("Normalized target path:", self.param1_input)
        self.layout.addRow("Source landmarks:", self.param2_input)
        self.layout.addRow("Target landmarks:", self.param3_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.layout.addWidget(self.buttons)

    def getInputs(self):
        return self.param1_input.text(), self.param2_input.text(), self.param3_input.text()


class Method3Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Method 2 Parameters")

        self.layout = QFormLayout(self)

        self.param1_input = QLineEdit(self)
        self.param2_input = QLineEdit(self)
        self.param3_input = QLineEdit(self)
        
        self.layout.addRow("Normalized target path:", self.param1_input)
        self.layout.addRow("Source landmarks:", self.param2_input)
        self.layout.addRow("Target landmarks:", self.param3_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.layout.addWidget(self.buttons)

    def getInputs(self):
        return self.param1_input.text(), self.param2_input.text(), self.param3_input.text()


def main():
    app = QApplication(sys.argv)
    ex = StainRegistrationApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
