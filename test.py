import customtkinter as ctk

print("1. Starting")

ctk.set_appearance_mode("dark")
print("2. Appearance set")

app = ctk.CTk()
print("3. CTk created")

app.title("CustomTkinter Test")
app.geometry("600x400")

label = ctk.CTkLabel(
    app,
    text="CUSTOMTKINTER WORKS",
    font=("Arial", 28)
)

label.pack(expand=True)

print("4. Widgets created")
print("5. Starting mainloop")

app.mainloop()

print("6. Mainloop ended")