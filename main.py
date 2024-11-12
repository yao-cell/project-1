import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline
import customtkinter as ctk
from pathlib import Path
import webbrowser
from CTkToolTip import CTkToolTip

class AIImageGenerator:
    def __init__(self):
        self.root = ctk.CTk()
        self.setup_window()
        self.load_model()
        self.create_widgets()
        self.current_image = None

    def setup_window(self):
        self.root.title("AI Image Generator")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Schedule maximize after a brief delay to ensure it works
        self.root.after(10, lambda: self.root.wm_state('zoomed'))

    def load_model(self):
        try:
            model_id = "CompVis/stable-diffusion-v1-4"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                use_auth_token="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"
            )
            self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            exit()

    def create_widgets(self):
        # Create sidebar
        self.sidebar = ctk.CTkFrame(self.root, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        # Logo and title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="AI Image Generator", 
                                     font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(40, 10))
        
        # Prompt input
        self.prompt_frame = ctk.CTkFrame(self.sidebar)
        self.prompt_frame.grid(row=1, column=0, padx=20, pady=(20, 0), sticky="ew")
        
        self.prompt_label = ctk.CTkLabel(self.prompt_frame, text="Enter your prompt:",
                                       font=ctk.CTkFont(size=14))
        self.prompt_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        
        self.prompt_entry = ctk.CTkTextbox(self.prompt_frame, height=300)
        self.prompt_entry.grid(row=1, column=0, padx=10, pady=(10, 10), sticky="ew")
        
        # Image size selection
        self.size_label = ctk.CTkLabel(self.sidebar, text="Image Size:", 
                                     font=ctk.CTkFont(size=14))
        self.size_label.grid(row=2, column=0, padx=20, pady=(20, 0), sticky="w")
        
        self.size_var = ctk.StringVar(value="256x256")
        self.size_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["256x256", "512x512", "768x768", "960x960", "1200x1200"],
            variable=self.size_var
        )
        self.size_menu.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="ew")
        
        # Action buttons
        self.button_frame = ctk.CTkFrame(self.sidebar)
        self.button_frame.grid(row=4, column=0, padx=10, pady=20, sticky="ews")
        
        self.generate_btn = ctk.CTkButton(
            self.button_frame,
            text="Generate",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self.generate_image
        )
        self.generate_btn.pack(fill="x", padx=10, pady=5)
        
        self.save_btn = ctk.CTkButton(
            self.button_frame,
            text="Save Image",
            font=ctk.CTkFont(size=14),
            height=40,
            command=self.save_image,
            state="disabled"
        )
        self.save_btn.pack(fill="x", padx=10, pady=5)
        
        self.clear_btn = ctk.CTkButton(
            self.button_frame,
            text="Clear",
            font=ctk.CTkFont(size=14),
            height=40,
            command=self.clear_all,
            fg_color="transparent",
            border_width=2
        )
        self.clear_btn.pack(fill="x", padx=10, pady=5)
        
        # Main content area (image display)
        self.content_frame = ctk.CTkFrame(self.root)
        self.content_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Image display
        self.image_label = ctk.CTkLabel(
            self.content_frame,
            text="Generated image will appear here",
            font=ctk.CTkFont(size=14),
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(self.root)
        self.progress.grid(row=3, column=1, padx=20, pady=(0, 20), sticky="ew")
        self.progress.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.root,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=2, column=1, padx=20, pady=(0, 5), sticky="e")
        
        # Add tooltips
        CTkToolTip(self.generate_btn, message="Generate a new image based on the prompt")
        CTkToolTip(self.save_btn, message="Save the generated image")
        CTkToolTip(self.clear_btn, message="Clear the current image and prompt")
        CTkToolTip(self.size_menu, message="Select the output image size")

    def update_status(self, message, progress_value=None):
        self.status_label.configure(text=message)
        if progress_value is not None:
            self.progress.set(progress_value)
        self.root.update()

    def generate_image(self):
        prompt = self.prompt_entry.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("Input Error", "Please enter a prompt!")
            return
            
        self.update_status("Generating image...", 0.2)
        self.generate_btn.configure(state="disabled")
        
        try:
            # Get size from selection
            width, height = map(int, self.size_var.get().split('x'))
            
            with torch.no_grad():
                self.update_status("Processing prompt...", 0.4)
                image = self.pipe(
                    prompt,
                    width=width,
                    height=height
                ).images[0]
            
            self.update_status("Preparing display...", 0.8)
            
            # Resize image for display while maintaining aspect ratio
            display_size = (1200, 1200)  # Maximum display size
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            self.current_image = image
            self.save_btn.configure(state="normal")
            self.update_status("Image generated successfully!", 1.0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating image: {e}")
            self.update_status("Error generating image", 0)
        
        finally:
            self.generate_btn.configure(state="normal")

    def save_image(self):
        if self.current_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                self.current_image.save(file_path)
                self.update_status(f"Image saved to {Path(file_path).name}")
                messagebox.showinfo("Success", "Image saved successfully!")

    def clear_all(self):
        self.prompt_entry.delete("1.0", "end")
        self.image_label.configure(image=None, text="Generated image will appear here")
        self.current_image = None
        self.save_btn.configure(state="disabled")
        self.update_status("Ready", 0)

    def run(self):
        self.root.mainloop()

def main():
    app = AIImageGenerator()
    app.run()

if __name__ == "__main__":
    main()