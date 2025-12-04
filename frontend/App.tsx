import React, { useState } from 'react';
import { AdminDashboardComponent, CreatorStatusComponent } from './TelemetryComponents';
import { UserRole } from './telemetry_mapper';

// Styles
const styles = {
    appContainer: {
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '20px',
        fontFamily: 'Segoe UI, Roboto, Helvetica, Arial, sans-serif',
    },
    navBar: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '30px',
        padding: '15px',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
    },
    title: {
        margin: 0,
        color: '#2c3e50',
    },
    controls: {
        display: 'flex',
        gap: '10px',
        alignItems: 'center',
    },
    button: {
        padding: '8px 16px',
        cursor: 'pointer',
        backgroundColor: '#007bff',
        color: 'white',
        border: 'none',
        borderRadius: '4px',
        fontSize: '14px',
        fontWeight: 'bold' as const,
    },
    roleBadge: {
        padding: '4px 8px',
        borderRadius: '4px',
        backgroundColor: '#e9ecef',
        fontSize: '12px',
        fontWeight: 'bold' as const,
        color: '#495057',
    },
    formSection: {
        marginBottom: '30px',
        padding: '20px',
        backgroundColor: '#fff',
        border: '1px solid #e9ecef',
        borderRadius: '8px',
    },
    inputGroup: {
        marginBottom: '15px',
    },
    label: {
        display: 'block',
        marginBottom: '5px',
        fontWeight: 'bold' as const,
        color: '#495057',
    },
    input: {
        width: '100%',
        padding: '8px',
        borderRadius: '4px',
        border: '1px solid #ced4da',
        fontSize: '16px',
    },
    textarea: {
        width: '100%',
        padding: '8px',
        borderRadius: '4px',
        border: '1px solid #ced4da',
        fontSize: '16px',
        minHeight: '100px',
    },
    submitButton: {
        padding: '10px 20px',
        cursor: 'pointer',
        backgroundColor: '#28a745',
        color: 'white',
        border: 'none',
        borderRadius: '4px',
        fontSize: '16px',
        fontWeight: 'bold' as const,
    }
};

const App: React.FC = () => {
    // State Management
    const [telemetryData, setTelemetryData] = useState<any>(null);
    const [userRole, setUserRole] = useState<UserRole>(UserRole.Admin);
    const [loading, setLoading] = useState<boolean>(false);

    // Form State
    const [courseTitle, setCourseTitle] = useState("Introduction to Agentic AI");
    const [learningOutcomes, setLearningOutcomes] = useState("Understand the basics of AI agents.\nLearn how to implement telemetry.");

    // Role Switcher Handler
    const toggleRole = () => {
        setUserRole(prev => prev === UserRole.Admin ? UserRole.CourseCreator : UserRole.Admin);
    };

    // API Handler
    const handleGenerate = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setTelemetryData(null); // Reset previous data

        try {
            const outcomesList = learningOutcomes.split('\n').filter(line => line.trim() !== '');

            const response = await fetch('/api/create-course-agentic', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    course_title: courseTitle,
                    learning_outcomes: outcomesList,
                    skip_video: true, // Faster for demo
                    num_modules: 1,   // Limit scope for demo
                    num_lessons: 2
                }),
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const result = await response.json();
            console.log("API Result:", result);

            if (result.telemetry) {
                setTelemetryData(result.telemetry);
            } else {
                console.warn("No telemetry data in response");
            }

        } catch (error) {
            console.error("Failed to generate course:", error);
            alert("Failed to generate course. Check console for details.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={styles.appContainer}>
            {/* Navigation / Controls */}
            <div style={styles.navBar}>
                <h1 style={styles.title}>AutoGen Course Platform</h1>

                <div style={styles.controls}>
                    <span style={styles.roleBadge}>Current View: {userRole}</span>
                    <button style={styles.button} onClick={toggleRole}>
                        Switch to {userRole === UserRole.Admin ? 'Creator' : 'Admin'} View
                    </button>
                </div>
            </div>

            {/* Input Form */}
            <div style={styles.formSection}>
                <h3>Generate New Course</h3>
                <form onSubmit={handleGenerate}>
                    <div style={styles.inputGroup}>
                        <label style={styles.label}>Course Title</label>
                        <input
                            style={styles.input}
                            type="text"
                            value={courseTitle}
                            onChange={(e) => setCourseTitle(e.target.value)}
                            required
                        />
                    </div>
                    <div style={styles.inputGroup}>
                        <label style={styles.label}>Learning Outcomes (one per line)</label>
                        <textarea
                            style={styles.textarea}
                            value={learningOutcomes}
                            onChange={(e) => setLearningOutcomes(e.target.value)}
                            required
                        />
                    </div>
                    <button
                        type="submit"
                        style={{ ...styles.submitButton, opacity: loading ? 0.7 : 1 }}
                        disabled={loading}
                    >
                        {loading ? 'Generating...' : 'Generate Course'}
                    </button>
                </form>
            </div>

            {/* Main Content Area */}
            {loading && (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                    <h3>Generating Course...</h3>
                    {/* Show Creator Status while loading if in Creator mode? 
              Actually, we don't have real-time streaming yet, so we just show a spinner. 
              But we can show the CreatorStatusComponent with a "Processing" state if we had partial data.
              For now, simple loading text.
          */}
                    <p>Please wait while agents research, plan, and create content.</p>
                </div>
            )}

            {!loading && telemetryData && (
                <>
                    {/* Conditional Rendering based on Role */}
                    {userRole === UserRole.Admin && (
                        <AdminDashboardComponent telemetryData={telemetryData} />
                    )}

                    {userRole === UserRole.CourseCreator && (
                        <CreatorStatusComponent telemetryData={telemetryData} />
                    )}
                </>
            )}

            {!loading && !telemetryData && (
                <div style={{ textAlign: 'center', color: '#666' }}>
                    <p>Enter course details above and click Generate to see telemetry.</p>
                </div>
            )}
        </div>
    );
};

export default App;
