'use client';

import React, { useState } from 'react';
import Link from 'next/link';

interface DocSection {
  id: string;
  category: string;
  title: string;
  badge: string;
  content: string;
  table?: { headers: string[]; rows: string[][] };
}

const DOCS_DATA: DocSection[] = [
  {
    id: 'attendance',
    category: 'Academic Ordinances',
    title: 'Clause 4.2: Mandatory Attendance & Condonation Rules',
    badge: 'Official Ordinance',
    content: `Under Jharkhand Rai University Academic Regulations, students must maintain a minimum of 75% attendance in both theory lectures and practical laboratory sessions to be eligible for end-semester examinations (ESE).
    
Key Highlights:
• Minimum Threshold: 75% attendance is mandatory across all registered courses.
• Medical Condonation: The Hon'ble Vice-Chancellor may condone up to a maximum of 10% attendance (allowing eligibility at 65%) strictly for documented medical emergencies or official university representation.
• Medical Submission Deadline: Medical certificates with hospital discharge summaries must be submitted to the Dean's Office within 7 working days of resuming classes.
• Debarment: Students below 65% attendance cannot be condoned and are declared 'Debarred' from end-semester exams for that semester.`,
  },
  {
    id: 'grading',
    category: 'Examination & Evaluation',
    title: 'UGC CBCS 10-Point Grading System & SGPA Formula',
    badge: 'UGC CBCS Standard',
    content: `Jharkhand Rai University follows the standardized UGC Choice Based Credit System (CBCS) 10-point letter grading scale. Continuous Internal Assessment (CIA) accounts for 40% and End-Semester Examination (ESE) accounts for 60%. A student must independently score at least 40% in CIA and 40% in ESE to pass a course.`,
    table: {
      headers: ['Letter Grade', 'Grade Point', 'Percentage Range', 'Performance Description'],
      rows: [
        ['O', '10.0', '90% and above', 'Outstanding'],
        ['A+', '9.0', '80% to 89.9%', 'Excellent'],
        ['A', '8.0', '70% to 79.9%', 'Very Good'],
        ['B+', '7.0', '60% to 69.9%', 'Good'],
        ['B', '6.0', '55% to 59.9%', 'Above Average'],
        ['C', '5.0', '50% to 54.9%', 'Average'],
        ['P', '4.0', '40% to 49.9%', 'Pass'],
        ['F', '0.0', 'Below 40%', 'Fail / Backlog'],
      ],
    },
  },
  {
    id: 'sgpa-formula',
    category: 'Examination & Evaluation',
    title: 'Official SGPA & CGPA-to-Percentage Conversion Formulas',
    badge: 'Mathematical Standard',
    content: `1. Semester Grade Point Average (SGPA):
   SGPA = Σ (Course Credits × Grade Points Earned) / Σ (Total Registered Course Credits)

2. Cumulative Grade Point Average (CGPA):
   CGPA = Σ (SGPA × Total Semester Credits) / Σ (Total Credits Across All Semesters)

3. Official Percentage Conversion Formula:
   Equivalent Percentage (%) = (CGPA - 0.75) × 10

Example: A student with an SGPA of 8.50 has an equivalent percentage of:
(8.50 - 0.75) × 10 = 7.75 × 10 = 77.50%`,
  },
  {
    id: 'backlog-promotion',
    category: 'Examination & Evaluation',
    title: 'Ordinance Clause 6.4: Back-Paper Examination & Promotion Rules',
    badge: 'Promotion Criteria',
    content: `• Maximum Backlog Allowance: A student can carry forward a maximum of 4 back-papers to be eligible for provisional promotion to the subsequent academic year.
• Year-Back Policy: If a student accumulates 5 or more uncleared back-papers across previous semesters, they are placed on Year-Back status and must clear back-papers during supplementary cycles.
• Supplementary Examinations: Held twice annually following odd and even semester examination announcements.`,
  },
  {
    id: 'fee-structure',
    category: 'Finance & Admissions',
    title: 'Institutional Tuition Fee Schedule (Per Semester)',
    badge: 'Finance Approved',
    content: `Tuition fees must be deposited before the commencement of classes each semester through the official ERP portal or university account. Late registration incurs a penalty of Rs. 50/day for the first 10 days, and Rs. 100/day thereafter up to Rs. 2,000.`,
    table: {
      headers: ['Degree Program', 'Duration', 'Tuition Fee / Sem', 'Exam Fee / Sem'],
      rows: [
        ['B.Tech Computer Science & Engg', '4 Years (8 Sem)', 'Rs. 65,000', 'Rs. 2,500'],
        ['B.Tech Mining Engineering', '4 Years (8 Sem)', 'Rs. 65,000', 'Rs. 2,500'],
        ['B.Tech Mechanical / Civil', '4 Years (8 Sem)', 'Rs. 60,000', 'Rs. 2,500'],
        ['Bachelor of Computer Applications (BCA)', '3 Years (6 Sem)', 'Rs. 30,000', 'Rs. 2,000'],
        ['Master of Computer Applications (MCA)', '2 Years (4 Sem)', 'Rs. 40,000', 'Rs. 2,500'],
        ['Bachelor of Pharmacy (B.Pharm)', '4 Years (8 Sem)', 'Rs. 55,000', 'Rs. 2,500'],
        ['Diploma in Pharmacy (D.Pharm)', '2 Years', 'Rs. 45,000', 'Rs. 2,000'],
        ['Bachelor of Business Admin (BBA)', '3 Years (6 Sem)', 'Rs. 30,000', 'Rs. 2,000'],
        ['Master of Business Admin (MBA)', '2 Years (4 Sem)', 'Rs. 60,000', 'Rs. 2,500'],
        ['Hostel & Mess Facility', 'Per Semester', 'Rs. 35,000', 'N/A'],
        ['Campus Bus Transportation', 'Per Semester', 'Rs. 8,000 - 12,000', 'N/A'],
      ],
    },
  },
  {
    id: 'scholarships',
    category: 'Finance & Admissions',
    title: 'Jharkhand E-Kalyan Post-Matric State Scholarship',
    badge: 'Govt. Scholarship',
    content: `Eligible SC, ST, and OBC students residing in Jharkhand can claim full or partial tuition reimbursement under the Jharkhand Government E-Kalyan scheme.
    
Eligibility Requirements:
1. Annual Family Income Ceiling: Below Rs. 2,50,000 for SC/ST, and below Rs. 1,50,000 for BC/EBC.
2. Required Documentation: Current Financial Year Income Certificate (SDO/CO level), Residential/Domicile Certificate, Caste Certificate, Previous Year Marksheet, JRU Bonafide Certificate with Fee Breakdown, and Aadhaar-seeded Bank Passbook.
3. Submission: Submit the online acknowledgement printout to the Dean of Student Welfare (DSW) office within 15 days of online application for institutional approval.`,
  },
];

export default function DocsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  const categories = ['All', 'Academic Ordinances', 'Examination & Evaluation', 'Finance & Admissions'];

  const filteredDocs = DOCS_DATA.filter((doc) => {
    const matchesCategory = selectedCategory === 'All' || doc.category === selectedCategory;
    const matchesSearch =
      doc.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      doc.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
      doc.category.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-main, #0b0f17)', color: 'var(--text-main, #f0f4f8)', fontFamily: 'sans-serif' }}>
      {/* Header */}
      <header style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', padding: '20px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <Link href="/" style={{ textDecoration: 'none', color: '#ea580c', fontWeight: 800, fontSize: '20px', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 32, height: 32, borderRadius: 8, background: 'linear-gradient(135deg, #c2410c, #f59e0b)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}>J</span>
            Jruva Docs
          </Link>
          <span style={{ fontSize: '13px', background: 'rgba(234,88,12,0.15)', color: '#fb923c', padding: '3px 10px', borderRadius: 12, border: '1px solid rgba(234,88,12,0.3)' }}>
            Institutional Knowledge Base
          </span>
        </div>
        <Link href="/" style={{ fontSize: '14px', color: '#94a3b8', textDecoration: 'none', border: '1px solid rgba(255,255,255,0.12)', padding: '6px 14px', borderRadius: 8 }}>
          ← Back to Chat
        </Link>
      </header>

      {/* Main Container */}
      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '40px 24px' }}>
        <div style={{ marginBottom: 36, textAlign: 'center' }}>
          <h1 style={{ fontSize: '32px', fontWeight: 800, marginBottom: 12, letterSpacing: '-0.5px' }}>
            Jharkhand Rai University Statutes & Ordinances
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '16px', maxWidth: 650, margin: '0 auto' }}>
            Authoritative guidelines, CBCS examination formulas, attendance rules, and fee structures available in Jruva.
          </p>
        </div>

        {/* Search & Category Filter */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 32 }}>
          <input
            type="text"
            placeholder="🔍 Search attendance, SGPA formula, fee structure, backlogs, scholarships..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ width: '100%', padding: '14px 20px', borderRadius: 12, border: '1px solid rgba(255,255,255,0.15)', background: 'rgba(15,23,42,0.6)', color: '#fff', fontSize: '15px', outline: 'none' }}
          />

          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {categories.map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setSelectedCategory(cat)}
                style={{
                  padding: '8px 16px',
                  borderRadius: 20,
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  border: selectedCategory === cat ? '1px solid #ea580c' : '1px solid rgba(255,255,255,0.1)',
                  background: selectedCategory === cat ? 'rgba(234,88,12,0.2)' : 'rgba(255,255,255,0.03)',
                  color: selectedCategory === cat ? '#fb923c' : '#94a3b8',
                }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Document Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {filteredDocs.map((doc) => (
            <section
              key={doc.id}
              id={doc.id}
              style={{
                background: 'rgba(15, 23, 42, 0.5)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 16,
                padding: '28px',
                backdropFilter: 'blur(8px)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                <div>
                  <span style={{ fontSize: '12px', textTransform: 'uppercase', color: '#ea580c', fontWeight: 700, letterSpacing: '0.5px' }}>
                    {doc.category}
                  </span>
                  <h2 style={{ fontSize: '20px', fontWeight: 700, marginTop: 4, color: '#f8fafc' }}>
                    {doc.title}
                  </h2>
                </div>
                <span style={{ fontSize: '11px', background: 'rgba(16,185,129,0.15)', color: '#34d399', padding: '4px 10px', borderRadius: 12, border: '1px solid rgba(16,185,129,0.3)', fontWeight: 600 }}>
                  {doc.badge}
                </span>
              </div>

              <p style={{ color: '#cbd5e1', fontSize: '14px', whiteSpace: 'pre-line', lineHeight: '1.7', marginBottom: doc.table ? 20 : 0 }}>
                {doc.content}
              </p>

              {doc.table && (
                <div style={{ overflowX: 'auto', marginTop: 16 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                    <thead>
                      <tr>
                        {doc.table.headers.map((h, i) => (
                          <th key={i} style={{ textAlign: 'left', padding: '10px 14px', background: 'rgba(234,88,12,0.12)', color: '#fb923c', borderBottom: '1px solid rgba(234,88,12,0.2)' }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {doc.table.rows.map((row, rIdx) => (
                        <tr key={rIdx} style={{ background: rIdx % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          {row.map((cell, cIdx) => (
                            <td key={cIdx} style={{ padding: '10px 14px', color: '#e2e8f0' }}>
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}
