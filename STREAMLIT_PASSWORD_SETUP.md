# 🔐 Dashboard Password Setup Guide

This guide explains how to configure password authentication for your Streamlit dashboard.

---

## 📋 **What Has Been Added**

Your dashboard now requires a password to access. The authentication system:
- ✅ Uses SHA-256 hashing for security (passwords are never stored in plain text)
- ✅ Supports Streamlit Cloud secrets management
- ✅ Has a fallback default password for local development
- ✅ Shows clear error messages for incorrect passwords
- ✅ Uses session state (password only needs to be entered once per session)

---

## 🏠 **For Local Development**

### Option 1: Use Default Password (Easiest)
The dashboard will automatically use password `demo123` if no secrets file is found.

```bash
# Just run the dashboard
streamlit run src/03_dashboard/dashboard.py

# Enter password: demo123
```

### Option 2: Configure Custom Password
1. Create/edit `.streamlit/secrets.toml`:
```toml
dashboard_password = "your_secure_password_here"
```

2. Run the dashboard:
```bash
streamlit run src/03_dashboard/dashboard.py

# Enter your custom password
```

**Note:** The `.streamlit/secrets.toml` file is already in `.gitignore` and will NOT be committed to GitHub (for security).

---

## ☁️ **For Streamlit Cloud (Production)**

### Step-by-Step Instructions:

#### 1️⃣ **Go to Your Streamlit Cloud Dashboard**
- Navigate to: https://share.streamlit.io/
- Log in with your GitHub account
- Find your deployed app

#### 2️⃣ **Open App Settings**
- Click on your app (Data-engineering dashboard)
- Click the **⋮** (three dots) menu in the top-right
- Select **"Settings"**

#### 3️⃣ **Add Secrets**
- In the left sidebar, click **"Secrets"**
- You'll see a text editor

#### 4️⃣ **Add Your Password**
Copy and paste this into the secrets editor:

```toml
dashboard_password = "YourSecurePassword123!"
```

**⚠️ Important:**
- Replace `YourSecurePassword123!` with your actual password
- Use a **strong password** (mix of letters, numbers, symbols)
- Don't share this password publicly!

#### 5️⃣ **Save Secrets**
- Click **"Save"** at the bottom
- Streamlit will automatically restart your app

#### 6️⃣ **Test the Password**
- Visit your app URL: `https://your-username-data-engineering.streamlit.app`
- You should see a password login screen
- Enter your password
- ✅ Access granted!

---

## 🔒 **Security Best Practices**

### ✅ **DO:**
- Use a strong, unique password for production
- Change the default password (`demo123`) on Streamlit Cloud
- Keep secrets file (`.streamlit/secrets.toml`) out of version control
- Use different passwords for different environments

### ❌ **DON'T:**
- Don't commit secrets to GitHub
- Don't share your password in Slack/email
- Don't use simple passwords like `password123`
- Don't hardcode passwords in your Python code

---

## 📸 **Visual Guide for Streamlit Cloud**

### Step 1: Find Your App
```
Streamlit Cloud Dashboard
├── Your apps
│   └── Data-engineering ← Click here
```

### Step 2: Open Settings
```
App Page (top-right)
└── ⋮ (three dots) → Settings
```

### Step 3: Add Secrets
```
Settings Sidebar
├── General
├── Secrets ← Click here
└── Advanced
```

### Step 4: Enter Password Secret
```toml
# Paste this into the editor:
dashboard_password = "YourStrongPassword!"
```

---

## 🧪 **Testing**

### Test Locally:
```bash
# Terminal 1: Start dashboard
streamlit run src/03_dashboard/dashboard.py

# Browser: http://localhost:8501
# ✅ Should show password screen
# ✅ Enter password: demo123 (or your custom password)
# ✅ Should show full dashboard
```

### Test on Streamlit Cloud:
1. Visit your deployed URL
2. Should see: "🔐 Dashboard Login"
3. Enter password configured in Cloud secrets
4. Should see full dashboard

---

## 🔄 **Changing the Password**

### Locally:
1. Edit `.streamlit/secrets.toml`
2. Change `dashboard_password = "new_password"`
3. Restart Streamlit
4. Use new password

### On Streamlit Cloud:
1. Go to app Settings → Secrets
2. Change the password value
3. Click "Save"
4. App restarts automatically
5. Use new password

---

## ❓ **Troubleshooting**

### "Incorrect password" error
- ✅ Check for typos
- ✅ Passwords are case-sensitive
- ✅ Make sure you saved the secrets file/Cloud secrets

### Password not working on Streamlit Cloud
- ✅ Check that secrets were saved correctly
- ✅ Try rebooting the app (Settings → Reboot)
- ✅ Make sure there are no extra spaces in the password

### Local development: "Using default password" warning
- ✅ This is normal if no `.streamlit/secrets.toml` exists
- ✅ Create the file to remove the warning
- ✅ Or ignore it (it's just a reminder)

---

## 📚 **Additional Resources**

- Streamlit Secrets Management: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- Streamlit Session State: https://docs.streamlit.io/library/api-reference/session-state

---

## 🎯 **Quick Reference**

| Environment | Password Location | Default Password |
|-------------|-------------------|------------------|
| **Local** | `.streamlit/secrets.toml` | `demo123` |
| **Streamlit Cloud** | App Settings → Secrets | Set manually |

**Current Status:**
- ✅ Code deployed to GitHub (`final-changes` branch)
- ✅ Local secrets file created (`.streamlit/secrets.toml`)
- ✅ Secrets file added to `.gitignore`
- ⏳ **Next step:** Configure password on Streamlit Cloud

---

## 📞 **Need Help?**

If you encounter issues:
1. Check that you're on the latest code (`git pull origin final-changes`)
2. Verify secrets are properly formatted (TOML syntax)
3. Check Streamlit Cloud logs (Settings → Logs)
4. Test locally first before deploying

---

**Created:** April 30, 2026  
**Last Updated:** April 30, 2026
