    /**
 * Invoice calculation utilities
 */

export function calculateSubtotal(items) {
  return items.reduce((sum, item) => sum + (item.quantity * item.price), 0);
}

export function calculateTax(subtotal, taxRate) {
  return (subtotal * taxRate) / 100;
}

export function calculateDiscount(amount, discountPercent, discountAmount) {
  if (discountPercent) {
    return (amount * discountPercent) / 100;
  }
  return discountAmount || 0;
}

export function calculateTotal(
  subtotal,
  taxAmount,
  discountAmount,
  additionalCharges = 0
) {
  return subtotal + taxAmount + additionalCharges - discountAmount;
}

export function formatCurrency(amount, currency = "₹") {
  return `${currency}${amount.toFixed(2)}`;
}

export function formatDate(date) {
  const d = typeof date === "string" ? new Date(date) : date;

  return new Intl.DateTimeFormat("en-IN", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(d);
}

export function getStatusColor(status) {
  const colors = {
    paid: "bg-green-100 text-green-700",
    pending: "bg-yellow-100 text-yellow-700",
    partial: "bg-blue-100 text-blue-700",
    cancelled: "bg-red-100 text-red-700",
    draft: "bg-gray-100 text-gray-700",
  };

  return colors[status] || colors.draft;
}

export function getStatusLabel(status) {
  const labels = {
    paid: "Paid",
    pending: "Pending",
    partial: "Partial Payment",
    cancelled: "Cancelled",
    draft: "Draft",
  };

  return labels[status] || status;
}

export function isRecentInvoice(date, days = 7) {
  const invoiceDate = typeof date === "string" ? new Date(date) : date;

  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days);

  return invoiceDate >= cutoffDate;
}

export function filterInvoices(invoices, filters) {
  let filtered = [...invoices];

  // Search filter
  if (filters.search) {
    const query = filters.search.toLowerCase();

    filtered = filtered.filter(
      (inv) =>
        inv.customer_name?.toLowerCase().includes(query) ||
        inv.id?.toString().includes(query) ||
        inv.customer_phone?.includes(query)
    );
  }

  // Status filter
  if (filters.status && filters.status !== "all") {
    filtered = filtered.filter((inv) => inv.status === filters.status);
  }

  // Date range filter
  if (filters.range && filters.range !== "all") {
    const daysMap = {
      last7: 7,
      last30: 30,
      last90: 90,
    };

    const days = daysMap[filters.range] || 30;

    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - days);

    filtered = filtered.filter(
      (inv) => new Date(inv.created_at) >= cutoffDate
    );
  }

  // Sort recent first
  filtered.sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  return filtered;
}

export function groupInvoicesByDate(invoices) {
  const groups = {
    recent: [],
    older: [],
  };

  invoices.forEach((invoice) => {
    if (isRecentInvoice(invoice.created_at, 7)) {
      groups.recent.push(invoice);
    } else {
      groups.older.push(invoice);
    }
  });

  return groups;
}

export function calculateInvoiceStats(invoices) {
  return {
    total: invoices.length,

    paid: invoices.filter((i) => i.status === "paid").length,

    pending: invoices.filter((i) => i.status === "pending").length,

    partial: invoices.filter((i) => i.status === "partial").length,

    totalAmount: invoices.reduce((sum, i) => sum + i.total, 0),

    paidAmount: invoices
      .filter((i) => i.status === "paid")
      .reduce((sum, i) => sum + i.total, 0),
  };
}