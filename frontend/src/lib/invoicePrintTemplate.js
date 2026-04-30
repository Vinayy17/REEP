const COMPANY = {
  name: "R RIDE",
  tagline: "BIKE GARAGE & STUDIO",
  address:
    "Mahaveer Building, gala no 09, near payal talkies, mandai road, Dhamankar naka, Bhiwandi, 421305",
  contact: "Phone: +91 90822 582580 | Email: outransystems@gmail.com",
  signatory: "R RIDE GARAGE",
  terms: [
    "Payment is due within 30 days of invoice date.",
    "Late payments may incur additional charges.",
    "Goods once sold will not be taken back or exchanged.",
    "All disputes subject to local jurisdiction only.",
  ],
}

const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")

const formatInvoiceDate = (value) => {
  if (!value) return "-"

  try {
    return new Date(value).toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    })
  } catch {
    return String(value)
  }
}

const formatMoney = (value, formatter) => `Rs. ${formatter(value || 0)}`

const getStatusMeta = (status) => {
  const normalized = String(status || "pending").toLowerCase()

  if (normalized === "paid") {
    return { label: "PAID", className: "badge-paid" }
  }

  if (normalized === "partial") {
    return { label: "PARTIAL", className: "badge-partial" }
  }

  if (normalized === "cancelled") {
    return { label: "CANCELLED", className: "badge-cancelled" }
  }

  return { label: "PENDING", className: "badge-pending" }
}

const buildProductMarkup = (item) => {
  const productName = escapeHtml(item?.product_name || "Item")
  const variantInfo = item?.variant_info || {}
  const details = []

  const sku = item?.sku || variantInfo?.v_sku
  if (sku) details.push(`SKU: ${sku}`)
  if (variantInfo?.variant_name) details.push(`Variant: ${variantInfo.variant_name}`)
  if (variantInfo?.color) details.push(`Color: ${variantInfo.color}`)
  if (variantInfo?.size) details.push(`Size: ${variantInfo.size}`)

  const detailsMarkup =
    details.length > 0
      ? `<div class="product-meta">${escapeHtml(details.join(" | "))}</div>`
      : ""

  return `<div class="product-name">${productName}</div>${detailsMarkup}`
}

export const buildInvoicePrintHtml = ({ invoice, subtotal, gstAmount, total, formatCurrency }) => {
  const statusMeta = getStatusMeta(invoice?.payment_status)
  const additionalCharges = Array.isArray(invoice?.additional_charges)
    ? invoice.additional_charges.filter((charge) => Number(charge?.amount || 0) > 0)
    : []

  const itemRows = (invoice?.items || [])
    .map((item, index) => {
      const lineTotal =
        item?.total !== undefined && item?.total !== null
          ? Number(item.total || 0)
          : Number(item?.quantity || 0) * Number(item?.price || 0)

      return `
        <tr>
          <td>${index + 1}</td>
          <td>${buildProductMarkup(item)}</td>
          <td class="text-right">${escapeHtml(item?.quantity || 0)}</td>
          <td class="text-right">${formatMoney(item?.price, formatCurrency)}</td>
          <td class="text-right">${escapeHtml(invoice?.gst_enabled ? `${item?.gst_rate ?? invoice?.gst_rate ?? 0}%` : "0%")}</td>
          <td class="text-right">${formatMoney(lineTotal, formatCurrency)}</td>
        </tr>
      `
    })
    .join("")

  const chargeRows = additionalCharges
    .map(
      (charge) => `
        <tr>
          <td>${escapeHtml(charge?.label || "Additional Charge")}</td>
          <td class="text-right">${formatMoney(charge?.amount, formatCurrency)}</td>
        </tr>
      `
    )
    .join("")

  const paymentRows = [
    Number(invoice?.advance_used || 0) > 0
      ? `
        <tr>
          <td>Advance Used</td>
          <td class="text-right">${formatMoney(invoice?.advance_used, formatCurrency)}</td>
        </tr>
      `
      : "",
    Number(invoice?.paid_amount || 0) > 0
      ? `
        <tr>
          <td>Paid Amount</td>
          <td class="text-right">${formatMoney(invoice?.paid_amount, formatCurrency)}</td>
        </tr>
      `
      : "",
    Number(invoice?.balance_amount || 0) > 0
      ? `
        <tr class="balance-row">
          <td>Balance Due</td>
          <td class="text-right">${formatMoney(invoice?.balance_amount, formatCurrency)}</td>
        </tr>
      `
      : "",
  ]
    .filter(Boolean)
    .join("")

  return `
    <!DOCTYPE html>
    <html>
      <head>
        <title>Invoice ${escapeHtml(invoice?.invoice_number || "")}</title>
        <meta charset="utf-8" />
        <style>
          * { box-sizing: border-box; }
          body {
            margin: 0;
            padding: 28px;
            font-family: Arial, sans-serif;
            background: #eef2f7;
            color: #0f172a;
          }
          .sheet {
            max-width: 920px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 28px;
            padding: 28px;
            box-shadow: 0 28px 80px rgba(15, 23, 42, 0.12);
          }
          .topbar {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            align-items: flex-start;
            padding-bottom: 18px;
            border-bottom: 2px solid #0f172a;
          }
          .brand-name {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: 0.04em;
          }
          .brand-tagline {
            font-size: 13px;
            font-weight: 700;
            color: #475569;
            margin-top: 4px;
            letter-spacing: 0.18em;
          }
          .brand-copy {
            font-size: 12px;
            line-height: 1.6;
            color: #475569;
            margin-top: 10px;
            max-width: 420px;
          }
          .invoice-panel {
            text-align: right;
          }
          .invoice-label {
            font-size: 11px;
            letter-spacing: 0.2em;
            color: #64748b;
            text-transform: uppercase;
          }
          .invoice-number {
            font-size: 22px;
            font-weight: 800;
            margin-top: 8px;
          }
          .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.12em;
            margin-top: 12px;
          }
          .badge-paid { background: #dcfce7; color: #166534; }
          .badge-partial { background: #fef3c7; color: #b45309; }
          .badge-pending { background: #fee2e2; color: #b91c1c; }
          .badge-cancelled { background: #e2e8f0; color: #334155; }
          .meta-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 18px;
            margin-top: 20px;
          }
          .meta-card {
            border: 1px solid #dbe4ee;
            border-radius: 22px;
            padding: 18px;
            background: #f8fafc;
          }
          .meta-title {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 10px;
          }
          .meta-card p {
            margin: 0;
            font-size: 13px;
            line-height: 1.7;
          }
          .meta-card strong {
            font-size: 15px;
          }
          .items-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 22px;
            border: 1px solid #dbe4ee;
            border-radius: 20px;
            overflow: hidden;
          }
          .items-table thead {
            background: #0f172a;
            color: #ffffff;
          }
          .items-table th,
          .items-table td {
            padding: 12px 14px;
            font-size: 12px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
          }
          .items-table tbody tr:nth-child(even) {
            background: #f8fafc;
          }
          .product-name {
            font-size: 13px;
            font-weight: 700;
            color: #0f172a;
          }
          .product-meta {
            margin-top: 4px;
            font-size: 10px;
            color: #64748b;
            line-height: 1.5;
          }
          .text-right { text-align: right; }
          .totals-wrap {
            display: flex;
            justify-content: flex-end;
            margin-top: 20px;
          }
          .totals-card {
            width: 320px;
            border: 1px solid #dbe4ee;
            border-radius: 22px;
            overflow: hidden;
            background: #ffffff;
          }
          .totals-card table {
            width: 100%;
            border-collapse: collapse;
          }
          .totals-card td {
            padding: 10px 14px;
            font-size: 12px;
            border-bottom: 1px solid #e2e8f0;
          }
          .totals-card tr:last-child td {
            border-bottom: none;
          }
          .total-row td {
            background: #ecfdf5;
            font-size: 15px;
            font-weight: 800;
            color: #166534;
          }
          .balance-row td {
            background: #fef2f2;
            font-weight: 800;
            color: #b91c1c;
          }
          .footer {
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 18px;
            margin-top: 26px;
            align-items: stretch;
          }
          .footer-card {
            border: 1px solid #dbe4ee;
            border-radius: 22px;
            padding: 18px;
            background: #ffffff;
          }
          .terms-title {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 12px;
          }
          .terms-list {
            margin: 0;
            padding-left: 18px;
            color: #475569;
            font-size: 12px;
            line-height: 1.7;
          }
          .signatures {
            display: flex;
            height: 100%;
            gap: 12px;
            align-items: flex-end;
            justify-content: space-between;
          }
          .signature-box {
            flex: 1;
            text-align: center;
            font-size: 11px;
            color: #475569;
          }
          .signature-line {
            border-top: 1px solid #94a3b8;
            margin-top: 52px;
            padding-top: 10px;
          }
          .signatory-name {
            margin-top: 8px;
            font-weight: 700;
            color: #0f172a;
          }
          .action-bar {
            text-align: center;
            margin-top: 20px;
          }
          .action-btn {
            border: none;
            border-radius: 999px;
            padding: 12px 20px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            margin: 0 6px;
          }
          .primary-btn {
            background: #0f172a;
            color: #ffffff;
          }
          .secondary-btn {
            background: #e2e8f0;
            color: #0f172a;
          }
          @media (max-width: 768px) {
            body { padding: 12px; }
            .sheet { padding: 18px; border-radius: 22px; }
            .topbar,
            .footer,
            .meta-grid { display: block; }
            .invoice-panel { text-align: left; margin-top: 18px; }
            .meta-card + .meta-card { margin-top: 14px; }
            .footer-card + .footer-card { margin-top: 14px; }
            .totals-card { width: 100%; }
            .items-table th,
            .items-table td { padding: 10px; }
          }
          @media print {
            body { padding: 0; background: #ffffff; }
            .sheet { box-shadow: none; border-radius: 0; border: none; max-width: none; }
            .action-bar { display: none; }
          }
        </style>
      </head>
      <body>
        <div class="sheet">
          <div class="topbar">
            <div>
              <div class="brand-name">${escapeHtml(COMPANY.name)}</div>
              <div class="brand-tagline">${escapeHtml(COMPANY.tagline)}</div>
              <div class="brand-copy">
                <div>${escapeHtml(COMPANY.address)}</div>
                <div>${escapeHtml(COMPANY.contact)}</div>
              </div>
            </div>

            <div class="invoice-panel">
              <div class="invoice-label">Invoice Details</div>
              <div class="invoice-number">${escapeHtml(invoice?.invoice_number || "-")}</div>
              <div class="badge ${statusMeta.className}">${statusMeta.label}</div>
              <div class="brand-copy" style="margin-top: 12px; text-align: inherit;">
                <div><strong>Date:</strong> ${escapeHtml(formatInvoiceDate(invoice?.created_at))}</div>
                <div><strong>Created By:</strong> ${escapeHtml(invoice?.created_by || "-")}</div>
                <div><strong>Mode:</strong> ${escapeHtml(String(invoice?.payment_mode || "cash").toUpperCase())}</div>
              </div>
            </div>
          </div>

          <div class="meta-grid">
            <div class="meta-card">
              <div class="meta-title">Invoice To</div>
              <p><strong>${escapeHtml(invoice?.customer_name || "Walk-in Customer")}</strong></p>
              <p>${escapeHtml(invoice?.customer_phone || "-")}</p>
              <p>${escapeHtml(invoice?.customer_address || "-")}</p>
            </div>
            <div class="meta-card">
              <div class="meta-title">Payment Snapshot</div>
              <p><strong>Cash Paid:</strong> ${formatMoney(invoice?.paid_amount || 0, formatCurrency)}</p>
              <p><strong>Advance Used:</strong> ${formatMoney(invoice?.advance_used || 0, formatCurrency)}</p>
              <p><strong>Balance:</strong> ${formatMoney(invoice?.balance_amount || 0, formatCurrency)}</p>
            </div>
          </div>

          <table class="items-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Product</th>
                <th class="text-right">Qty</th>
                <th class="text-right">Price</th>
                <th class="text-right">${invoice?.gst_enabled ? `GST ${escapeHtml(invoice?.gst_rate ?? 0)}%` : "GST"}</th>
                <th class="text-right">Amount</th>
              </tr>
            </thead>
            <tbody>
              ${itemRows}
            </tbody>
          </table>

          <div class="totals-wrap">
            <div class="totals-card">
              <table>
                <tr>
                  <td>Subtotal</td>
                  <td class="text-right">${formatMoney(subtotal, formatCurrency)}</td>
                </tr>
                ${chargeRows}
                ${
                  Number(gstAmount || 0) > 0
                    ? `
                      <tr>
                        <td>GST Amount</td>
                        <td class="text-right">${formatMoney(gstAmount, formatCurrency)}</td>
                      </tr>
                    `
                    : ""
                }
                ${
                  Number(invoice?.discount || 0) > 0
                    ? `
                      <tr>
                        <td>Discount</td>
                        <td class="text-right">- ${formatMoney(invoice?.discount, formatCurrency)}</td>
                      </tr>
                    `
                    : ""
                }
                <tr class="total-row">
                  <td>Total Amount</td>
                  <td class="text-right">${formatMoney(total, formatCurrency)}</td>
                </tr>
                ${paymentRows}
              </table>
            </div>
          </div>

          <div class="footer">
            <div class="footer-card">
              <div class="terms-title">Terms & Conditions</div>
              <ol class="terms-list">
                ${COMPANY.terms.map((term) => `<li>${escapeHtml(term)}</li>`).join("")}
              </ol>
            </div>

            <div class="footer-card">
              <div class="signatures">
                <div class="signature-box">
                  <div class="signature-line">Customer Signature</div>
                </div>
                <div class="signature-box">
                  <div class="signature-line">Authorized Signatory</div>
                  <div class="signatory-name">${escapeHtml(COMPANY.signatory)}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="action-bar">
          <button class="action-btn primary-btn" onclick="window.print()">Print Invoice</button>
          <button class="action-btn secondary-btn" onclick="window.close()">Close</button>
        </div>
      </body>
    </html>
  `
}
