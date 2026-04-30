# Complete Account Flow Explanation

## Overview
Your accounting system is a **double-entry bookkeeping system** that tracks all financial transactions (income, expenses, payments) and generates real-time financial reports like Ledger, Cashbook, Bankbook, GST Ledger, Expenses, and Profit/Loss statements.

---

## 🏗️ System Architecture

### Frontend (accounts.js)
- **Framework**: React (Next.js)
- **Purpose**: UI for viewing accounts, creating entries, and exporting data
- **Key Components**: Tabs, Forms, Cards, Tables, Export buttons

### Backend (main.py)
- **Framework**: FastAPI
- **Database**: MySQL/PostgreSQL with SQLAlchemy ORM
- **Purpose**: API endpoints for CRUD operations and calculations

---

## 📊 Database Models (Backend)

### Key Tables:

#### 1. **LedgerEntry** (Core Accounting Table)
```
- entry_type: invoice, expense, payment_in, payment_out
- debit: Money flowing out (expenses, purchases)
- credit: Money flowing in (income, sales)
- entry_date: Date of transaction
- customer_id / supplier_id: Who is involved
- payment_mode: cash, bank, cheque
- Tax: cgst, sgst, igst amounts
```
**Purpose**: Records EVERY financial transaction in double-entry format.

#### 2. **ExpenseModel**
```
- title: What expense is for
- amount: How much was spent
- category_id: Link to category
- payment_mode: How was it paid
- expense_date: When it happened
```
**Purpose**: Tracks all expenses separately.

#### 3. **ChequeModel**
```
- cheque_number: Cheque ID
- amount: Cheque amount
- party_name: Who it's for (customer/supplier)
- status: pending/cleared/bounced
```
**Purpose**: Tracks cheque payments separately.

#### 4. **CustomerModel & SupplierModel**
```
- current_balance: How much they owe (customers) or we owe them (suppliers)
- opening_balance: Initial balance when created
```
**Purpose**: Track party-wise balances.

#### 5. **InvoiceModel**
```
- invoice_number: Unique ID
- customer_id: Who bought
- items: Product details (JSON)
- subtotal, gst_amount, total: Calculated totals
- payment_status: pending/paid/partial
```
**Purpose**: Records sales transactions.

---

## 🔄 Complete Transaction Flow

### **Scenario 1: Customer Buys Products (Creates Invoice)**

#### Flow:
1. **Frontend (accounts.js)** - User creates invoice via "Invoice" button in Customers page
2. **API Call**: POST `/api/invoices`
3. **Backend Processing**:
   - Save invoice in `invoices` table
   - Create `LedgerEntry`:
     - **Debit**: Accounts Receivable (customer owes you)
     - **Credit**: Sales Revenue (you made income)
   - Update `CustomerModel.current_balance` (add to what they owe)
4. **Frontend Updates**: Fetch new ledger data and display

#### Backend Code (main.py):
```python
# Creates a ledger entry when invoice is created
POST @api_router.post("/invoices")
- entry_type = "invoice"
- debit = invoice.total (what customer owes)
- credit = invoice.subtotal (your income)
- customer_id = customer_id
```

---

### **Scenario 2: Customer Pays You (Payment In)**

#### Flow:
1. **Frontend (accounts.js)** - User clicks "Receive Payment"
2. **Form Input**:
   - Customer ID
   - Amount
   - Payment Mode (cash/bank/cheque)
   - Date
3. **API Call**: POST `/api/accounts/payment-in`
4. **Backend Processing**:
   - Create `LedgerEntry`:
     - **Debit**: Cash/Bank (you receive money)
     - **Credit**: Accounts Receivable (customer debt reduces)
   - Update `CustomerModel.current_balance` (subtract paid amount)
   - If cheque: Create `ChequeModel` record
5. **Frontend Updates**: Ledger refreshed, balances updated

#### Backend Code:
```python
POST @api_router.post("/accounts/payment-in")
- Check customer exists
- Create ledger entry with payment details
- Update customer balance
- If cheque, create cheque record
```

---

### **Scenario 3: You Pay Supplier (Payment Out)**

#### Flow:
1. **Frontend (accounts.js)** - User clicks "Pay Supplier"
2. **Form Input**:
   - Supplier ID
   - Amount
   - Payment Mode
   - Date
3. **API Call**: POST `/api/accounts/payment-out`
4. **Backend Processing**:
   - Create `LedgerEntry`:
     - **Debit**: Accounts Payable/Expense (you owe money)
     - **Credit**: Cash/Bank (money goes out)
   - Update `SupplierModel.current_balance`
   - If cheque: Create `ChequeModel`

---

### **Scenario 4: Record Expense (Add Expense)**

#### Flow:
1. **Frontend (accounts.js)** - User clicks "Add Expense"
2. **Form Input**:
   - Title (e.g., "Electricity Bill")
   - Amount
   - Category
   - Payment Mode
   - Date
3. **API Call**: POST `/api/expenses`
4. **Backend Processing**:
   - Save in `expenses` table
   - Create `LedgerEntry`:
     - **Debit**: Expense (money spent)
     - **Credit**: Cash/Bank (money from pocket)

---

## 📈 Statistics & Reports (The 4 Summary Cards)

### **1. Cash Balance**
```
= Total Money in Hand
= All cash transactions (cash payments in - cash payments out)
From: Ledger entries where payment_mode = "cash"
```

**Calculation** (Backend `/api/accounts/summary`):
```python
cash_balance = SUM(credit) - SUM(debit)
WHERE payment_mode = "cash"
```

### **2. Bank Balance**
```
= Total Money in Bank Account
= All bank transactions (bank deposits - bank withdrawals)
From: Ledger entries where payment_mode = "bank"
```

**Calculation**:
```python
bank_balance = SUM(credit) - SUM(debit)
WHERE payment_mode = "bank"
```

### **3. Total Income**
```
= All money coming IN from customers
From: Ledger entries with entry_type = "invoice" (credit side)
```

**Calculation**:
```python
total_income = SUM(credit)
WHERE entry_type = "invoice"
```

### **4. Total Expenses**
```
= All money going OUT for operations
From: Expense table + Ledger entries with entry_type = "expense" (debit side)
```

**Calculation**:
```python
total_expenses = SUM(amount) FROM expenses
+ SUM(debit) FROM ledger WHERE entry_type = "expense"
```

---

## 💰 Profit & Loss Calculation

```
Profit/Loss = Total Income - Total Expenses

Formula:
= Income from invoices
- Cost of goods/services
- All operating expenses
- All payments to suppliers

Result:
IF Profit > 0: Green card ✅ (Making money)
IF Profit < 0: Red card ❌ (Losing money)
```

**Backend Code**:
```python
@api_router.get("/accounts/profit-loss")
profit = total_income - total_expenses
return {
    "income": total_income,
    "expense": total_expenses,
    "profit": profit
}
```

---

## 📋 Ledger vs Cashbook vs Bankbook

| Feature | Ledger | Cashbook | Bankbook |
|---------|--------|----------|----------|
| **Shows** | All transactions | Only cash transactions | Only bank transactions |
| **Entry Types** | All (invoices, expenses, payments) | Cash payments only | Bank payments only |
| **Purpose** | Complete financial record | Track physical cash | Track bank account |
| **Used For** | Financial statements, audits | Cash counting, reconciliation | Bank reconciliation |

---

## 🧮 Double-Entry Bookkeeping Logic

Every transaction has TWO sides:

### Example 1: Invoice Created (₹1000)
| Account | Debit | Credit |
|---------|-------|--------|
| Accounts Receivable | ₹1000 | |
| Sales Revenue | | ₹1000 |
**Meaning**: Customer owes ₹1000, you earned ₹1000

### Example 2: Customer Pays (₹1000)
| Account | Debit | Credit |
|---------|-------|--------|
| Cash | ₹1000 | |
| Accounts Receivable | | ₹1000 |
**Meaning**: You got ₹1000 cash, customer debt reduced by ₹1000

### Example 3: Expense Paid (₹500)
| Account | Debit | Credit |
|---------|-------|--------|
| Expense | ₹500 | |
| Cash | | ₹500 |
**Meaning**: ₹500 spent on expense, cash reduced by ₹500

---

## 🔌 Frontend-Backend Data Flow (Step by Step)

### User creates an expense:
```
1. Frontend (accounts.js)
   ↓ User fills form + clicks "Add Expense"
   ↓ expenseTitle = "Power Bill"
   ↓ expenseAmount = 500
   ↓ expensePaymentMode = "cash"

2. API Request
   ↓ POST /api/expenses
   ↓ Headers: Authorization: Bearer {token}
   ↓ Body: {title, amount, category_id, payment_mode, expense_date}

3. Backend (main.py)
   ↓ Receives request in addExpense() function
   ↓ Validates: title and amount not empty
   ↓ Creates ExpenseModel record
   ↓ Creates LedgerEntry:
      - entry_type = "expense"
      - debit = 500 (money out)
      - credit = 0
      - payment_mode = "cash"

4. Database
   ↓ Saves expense record
   ↓ Saves ledger entry
   ↓ Updates cash_balance calculation

5. Frontend Response
   ↓ Toast: "Expense added successfully"
   ↓ Calls fetchAllData()
   ↓ Re-fetches summary, profit/loss, ledger
   ↓ Updates UI with new stats
```

---

## 🎯 Key Formulas Used

### Cash Balance:
```
= Σ(All credits with payment_mode="cash") - Σ(All debits with payment_mode="cash")
```

### Bank Balance:
```
= Σ(All credits with payment_mode="bank") - Σ(All debits with payment_mode="bank")
```

### Total Income:
```
= Σ(All invoice credits)
```

### Total Expenses:
```
= Σ(All expense debits) + Σ(All supplier payment debits)
```

### Customer Balance:
```
= opening_balance + Σ(invoices) - Σ(payments received)
```

### Supplier Balance:
```
= opening_balance + Σ(purchases) - Σ(payments made)
```

---

## 🔍 Tab-Specific Logic

### **Ledger Tab**
- Shows ALL transactions
- Includes debit/credit columns
- Filters by date range, payment mode
- Used for complete financial records

### **Cashbook Tab**
- Shows only cash transactions
- Calculates running cash balance
- Used for cash reconciliation

### **Bankbook Tab**
- Shows only bank transactions
- Calculates running bank balance
- Used for bank reconciliation

### **GST Ledger Tab**
- Shows GST breakup (CGST, SGST, IGST)
- Calculates total GST collected
- Used for GST filing

### **Expenses Tab**
- Shows all expenses
- Grouped by category
- Shows total expenses by category

### **Cheques Tab**
- Shows all cheque records
- Status: pending/cleared/bounced
- Used for cheque reconciliation

---

## 📤 Export Functionality

### Export Ledger:
```
GET /api/accounts/export/ledger?start_date=...&end_date=...
↓
Backend generates CSV with all ledger entries
↓
Frontend downloads as ledger_TIMESTAMP.csv
```

### Export Expenses:
```
GET /api/accounts/export/expenses?start_date=...&end_date=...
↓
Backend generates CSV with all expenses
↓
Frontend downloads as expenses_TIMESTAMP.csv
```

---

## 🔐 Security & Validation

1. **JWT Token**: All requests require `Authorization: Bearer {token}`
2. **User Verification**: Token decoded to verify user identity
3. **Input Validation**:
   - Title/Amount cannot be empty
   - Amount must be positive number
   - Dates must be valid
4. **Error Handling**: Returns proper error messages

---

## 🚀 Complete Transaction Example

**You run a furniture business. Here's a full day:**

### 9:00 AM - Customer buys ₹10,000 worth of furniture
```
API: POST /api/invoices
Ledger Entry:
  Debit: Accounts Receivable ₹10,000
  Credit: Sales Revenue ₹10,000
Customer Balance: Now owes ₹10,000
```

### 11:00 AM - Supplier gives you items, you owe ₹5,000
```
API: POST /api/accounts/payment-out
Ledger Entry:
  Debit: Accounts Payable ₹5,000
  Credit: Cash ₹5,000
Cash Balance: Reduced by ₹5,000
Supplier Balance: Now you owe ₹5,000
```

### 3:00 PM - Electricity bill comes, ₹1,000
```
API: POST /api/expenses
Ledger Entry:
  Debit: Expense ₹1,000
  Credit: Cash ₹1,000
Total Expenses: Increased by ₹1,000
Cash Balance: Reduced by ₹1,000
```

### 5:00 PM - Customer pays ₹10,000
```
API: POST /api/accounts/payment-in
Ledger Entry:
  Debit: Cash ₹10,000
  Credit: Accounts Receivable ₹10,000
Cash Balance: Increased by ₹10,000
Customer Balance: Now ₹0 (paid in full)
```

### **End of Day Summary Cards:**
- **Cash Balance**: ₹10,000 - ₹5,000 - ₹1,000 + ₹10,000 = **₹14,000**
- **Income**: ₹10,000 (from customer invoice)
- **Expenses**: ₹1,000 (electricity)
- **Profit**: ₹10,000 - ₹1,000 = **₹9,000** ✅

---

## 💡 Summary

Your system works like a real business accounting setup:

1. **Every transaction is recorded twice** (double-entry)
2. **Money flows are tracked** by type (cash, bank, cheque)
3. **Balances are auto-calculated** in real-time
4. **Reports are generated** from the ledger data
5. **Profit/Loss is computed** automatically
6. **Everything is auditable** with full transaction history

This is a **complete accounting system** used by real businesses! 🎯

