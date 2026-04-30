# Graphify View

This is a graph-style map of the current backend in [main.py](/D:/lights-main/b/main.py).

## System Map

```mermaid
flowchart TD
    A["FastAPI App"] --> B["Startup"]
    B --> B1["Create accounting tables"]
    B --> B2["Ensure Power Bill expense category"]
    B --> B3["Initialize Redis rate limiter"]

    A --> C["Auth"]
    C --> C1["POST /auth/register"]
    C --> C2["POST /auth/login"]

    A --> D["Master Data"]
    D --> D1["Categories"]
    D --> D2["Products"]
    D --> D3["Combos"]

    A --> E["Inventory"]
    E --> E1["Lookup by SKU"]
    E --> E2["Material Inward"]
    E --> E3["Material Outward"]
    E --> E4["Transactions"]
    E --> E5["Inventory Report PDF"]
    E --> E6["Ageing Report"]

    A --> F["Requirements"]
    F --> F1["Create"]
    F --> F2["List"]
    F --> F3["Complete / Reject"]
    F --> F4["Cleanup"]

    A --> G["Customers + Invoices"]
    G --> G1["Customers CRUD"]
    G --> G2["Customer Advance"]
    G --> G3["Draft Invoices"]
    G --> G4["Finalize Draft"]
    G --> G5["Final Invoices"]
    G --> G6["Invoice Payments"]
    G --> G7["Invoice Cancel / Status"]
    G --> G8["Invoice PDF Export"]

    A --> H["Accounting"]
    H --> H1["Expense Categories"]
    H --> H2["Expenses"]
    H --> H3["Employees"]
    H --> H4["Salary Payments"]
    H --> H5["Power Bills"]
    H --> H6["Suppliers"]
    H --> H7["Supplier Invoices"]
    H --> H8["Payment In / Out"]
    H --> H9["Ledger / Cashbook / Bankbook"]
    H --> H10["Trial Balance / GST / P&L"]
    H --> H11["Cheques"]
    H --> H12["Dashboard APIs"]
```

## Data Model Graph

```mermaid
erDiagram
    USERMODEL {
        string id
        string email
        string role
    }

    CUSTOMERMODEL {
        string id
        string name
        float current_balance
    }

    INVOICEMODEL {
        string id
        string invoice_number
        string customer_id
        float total
        float paid_amount
        float balance_amount
        string payment_status
    }

    INVOICEPAYMENT {
        string id
        string invoice_id
        float amount
        string payment_mode
    }

    LEDGERENTRY {
        string id
        string entry_type
        string customer_id
        string supplier_id
        float debit
        float credit
    }

    SUPPLIERMODEL {
        string id
        string name
        float current_balance
    }

    SUPPLIERINVOICEMODEL {
        string id
        string supplier_id
        float total_amount
        float paid_amount
        float pending_amount
        string status
    }

    EMPLOYEEMODEL {
        string id
        string name
        float salary
    }

    SALARYPAYMENTMODEL {
        string id
        string employee_id
        float total_salary
        float paid_amount
        float pending_amount
    }

    EXPENSECATEGORYMODEL {
        string id
        string name
    }

    EXPENSEMODEL {
        string id
        string category_id
        float amount
        string payment_mode
    }

    CATEGORYMODEL {
        string id
        string name
    }

    PRODUCTMODEL {
        string id
        string category_id
        string sku
        int stock
    }

    INVENTORYTRANSACTION {
        string id
        string product_id
        string type
        int quantity
    }

    COMBOMODEL {
        string id
        string name
        float price
    }

    CHEQUEMODEL {
        string id
        string cheque_number
        float amount
        string party_type
        string status
    }

    REQUIREMENTMODEL {
        string id
        string customer_name
        string status
    }

    CUSTOMERMODEL ||--o{ INVOICEMODEL : has
    INVOICEMODEL ||--o{ INVOICEPAYMENT : receives
    CUSTOMERMODEL ||--o{ LEDGERENTRY : posts
    SUPPLIERMODEL ||--o{ SUPPLIERINVOICEMODEL : has
    SUPPLIERMODEL ||--o{ LEDGERENTRY : posts
    EMPLOYEEMODEL ||--o{ SALARYPAYMENTMODEL : has
    EXPENSECATEGORYMODEL ||--o{ EXPENSEMODEL : groups
    CATEGORYMODEL ||--o{ PRODUCTMODEL : contains
    PRODUCTMODEL ||--o{ INVENTORYTRANSACTION : moves
```

## Core Balance Flow

```mermaid
flowchart LR
    A["Create Invoice"] --> B["calculate_invoice_settlement()"]
    B --> C["advance_used"]
    B --> D["cash_paid_amount"]
    B --> E["balance_amount"]
    E --> F["update_customer_balance(+pending)"]
    C --> G["update_customer_balance(+advance_used)"]
    E --> H["sync_invoice_pending_ledger()"]
    D --> I["InvoicePayment + sale_payment ledger"]

    J["Add Customer Advance"] --> K["update_customer_balance(-amount)"]
    K --> L["advance_in ledger"]

    M["Invoice Partial / Paid"] --> N["update_customer_balance(-received_amount)"]
    N --> O["sale_payment ledger"]
    N --> P["sync_invoice_pending_ledger()"]

    Q["Cancel Invoice"] --> R["Restore stock"]
    Q --> S["Reverse customer balance"]
    Q --> T["Delete invoice ledger entries"]
    Q --> U["Delete payment records"]
```

## API Groups By Area

```mermaid
mindmap
  root((main.py API))
    Auth
      register
      login
    Uploads
      product image
      variant image
      requirement image
    Requirements
      create
      list
      complete
      reject
      cleanup
    Catalog
      categories
      products
      combos
    Inventory
      inward
      outward
      lookup
      transactions
      report
      ageing
    Customers
      create
      list
      update
      delete
      search
      advance
    Invoices
      draft create
      draft update
      draft list
      draft finalize
      invoice create
      invoice list
      status update
      add payment
      complete payment
      payment delete
      pdf export
    Dashboard
      today
      low stock
      top products
      inventory movement
      activity
      hourly sales
      sales chart
    Accounting
      expense categories
      expenses
      employees
      salary payments
      power bills
      suppliers
      supplier invoices
      payment in
      payment out
      cashbook
      bankbook
      GST ledger
      profit and loss
      ledger
      trial balance
      cheques
```

## Route Cluster Reference

- Auth starts around [main.py](/D:/lights-main/b/main.py#L1082)
- Draft invoice flow starts around [main.py](/D:/lights-main/b/main.py#L2278)
- Requirements starts around [main.py](/D:/lights-main/b/main.py#L3196)
- Products and inventory start around [main.py](/D:/lights-main/b/main.py#L3345)
- Customer and invoice final flow starts around [main.py](/D:/lights-main/b/main.py#L4698)
- Dashboard APIs start around [main.py](/D:/lights-main/b/main.py#L6364)
- Accounting APIs start around [main.py](/D:/lights-main/b/main.py#L6801)

## Quick Notes

- Customer balance logic is centralized around `update_customer_balance()`, `calculate_invoice_settlement()`, and `sync_invoice_pending_ledger()`.
- This file contains two `/customers` GET routes, one earlier and one later, which is worth cleaning later because route order can make behavior confusing.
- `main.py` is acting as one large monolith right now: models, schemas, helpers, business rules, and routes all live together.
