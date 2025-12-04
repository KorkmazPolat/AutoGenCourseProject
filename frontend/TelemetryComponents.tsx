import React from 'react';
import { TelemetryMapper, UserRole, TelemetryData } from './telemetry_mapper';

// --- Styles (Inline for portability) ---
const styles = {
    container: {
        fontFamily: 'sans-serif',
        padding: '20px',
        border: '1px solid #ddd',
        borderRadius: '8px',
        marginBottom: '20px',
        backgroundColor: '#fff',
    },
    header: {
        borderBottom: '1px solid #eee',
        paddingBottom: '10px',
        marginBottom: '15px',
    },
    section: {
        marginBottom: '20px',
    },
    sectionTitle: {
        fontSize: '1.1em',
        fontWeight: 'bold',
        marginBottom: '10px',
        color: '#333',
    },
    // Admin Specific
    cardGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '10px',
        marginBottom: '15px',
    },
    statCard: {
        background: '#f9f9f9',
        padding: '10px',
        borderRadius: '4px',
        textAlign: 'center' as const,
    },
    statValue: {
        fontSize: '1.5em',
        fontWeight: 'bold',
        color: '#007bff',
    },
    table: {
        width: '100%',
        borderCollapse: 'collapse' as const,
        fontSize: '0.9em',
    },
    th: {
        textAlign: 'left' as const,
        padding: '8px',
        borderBottom: '2px solid #ddd',
    },
    td: {
        padding: '8px',
        borderBottom: '1px solid #eee',
    },
    barChart: {
        display: 'flex',
        alignItems: 'flex-end',
        height: '100px',
        gap: '5px',
        padding: '10px',
        background: '#f0f0f0',
        borderRadius: '4px',
    },
    bar: {
        background: '#dc3545',
        width: '20px',
        transition: 'height 0.3s',
    },
    // Creator Specific
    progressBarContainer: {
        height: '20px',
        background: '#e9ecef',
        borderRadius: '10px',
        overflow: 'hidden',
        marginBottom: '10px',
    },
    progressBarFill: (percent: number) => ({
        height: '100%',
        width: `${percent}%`,
        background: '#28a745',
        transition: 'width 0.5s ease-in-out',
    }),
    badge: {
        display: 'inline-block',
        padding: '5px 10px',
        borderRadius: '15px',
        background: '#17a2b8',
        color: 'white',
        fontSize: '0.9em',
        fontWeight: 'bold',
    },
    alertBox: {
        marginTop: '15px',
        padding: '10px',
        background: '#fff3cd',
        border: '1px solid #ffeeba',
        color: '#856404',
        borderRadius: '4px',
    },
};

interface Props {
    telemetryData: TelemetryData;
}

/**
 * Admin Dashboard: Full technical visibility.
 */
export const AdminDashboardComponent: React.FC<Props> = ({ telemetryData }) => {
    const view = TelemetryMapper.map(telemetryData, UserRole.Admin);

    if (!view) return <div>Access Denied</div>;

    const { errorGraphs, performanceHeatmaps, technicalQueue } = view;

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <h2>Admin Telemetry Dashboard</h2>
            </div>

            {/* 1. Failure Trends */}
            <div style={styles.section}>
                <div style={styles.sectionTitle}>Failure Trends (Rejections per Stage)</div>
                {Object.keys(errorGraphs).length > 0 ? (
                    <div style={styles.barChart}>
                        {Object.entries(errorGraphs).map(([stage, count]) => {
                            // Simple scaling for demo
                            const height = Math.min((count as number) * 20, 100);
                            return (
                                <div key={stage} title={`${stage}: ${count}`} style={{
                                    ...styles.bar,
                                    height: `${height}%`
                                }} />
                            );
                        })}
                    </div>
                ) : (
                    <div>No failures recorded.</div>
                )}
            </div>

            {/* 2. Performance Heatmap (Table View) */}
            <div style={styles.section}>
                <div style={styles.sectionTitle}>Performance Metrics</div>
                <table style={styles.table}>
                    <thead>
                        <tr>
                            <th style={styles.th}>Stage</th>
                            <th style={styles.th}>Avg (s)</th>
                            <th style={styles.th}>Max (s)</th>
                            <th style={styles.th}>Count</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.entries(performanceHeatmaps || {}).map(([stage, stats]: [string, any]) => (
                            <tr key={stage}>
                                <td style={styles.td}>{stage}</td>
                                <td style={styles.td}>{stats.avg_seconds.toFixed(2)}</td>
                                <td style={styles.td}>{stats.max_seconds.toFixed(2)}</td>
                                <td style={styles.td}>{stats.count}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* 3. Technical Queue */}
            <div style={styles.section}>
                <div style={styles.sectionTitle}>Workload Queue</div>
                <div style={styles.cardGrid}>
                    <div style={styles.statCard}>
                        <div>Active Tasks</div>
                        <div style={styles.statValue}>{technicalQueue.activeTasks}</div>
                    </div>
                    <div style={styles.statCard}>
                        <div>Max Depth</div>
                        <div style={styles.statValue}>{technicalQueue.queueDepth}</div>
                    </div>
                </div>

                <h4>Recent Tasks</h4>
                <ul style={{ fontSize: '0.9em', color: '#666' }}>
                    {technicalQueue.taskLog.slice(-5).map((task: any, idx: number) => (
                        <li key={idx}>
                            {task.task_name} - {task.success ? '✅' : '❌'} ({task.duration.toFixed(2)}s)
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

/**
 * Creator Status: Simplified, friendly view.
 */
export const CreatorStatusComponent: React.FC<Props> = ({ telemetryData }) => {
    const view = TelemetryMapper.map(telemetryData, UserRole.CourseCreator);

    if (!view) return <div>Access Denied</div>;

    const { progressBar, estimatedWaitTime, smartAlerts } = view;

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <h2>Course Generation Status</h2>
            </div>

            {/* 1. Progress Bar */}
            <div style={styles.section}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                    <strong>{progressBar?.status}</strong>
                    <span>{progressBar?.percent}%</span>
                </div>
                <div style={styles.progressBarContainer}>
                    <div style={styles.progressBarFill(progressBar?.percent || 0)} />
                </div>
            </div>

            {/* 2. Wait Time */}
            <div style={styles.section}>
                <span style={styles.badge}>
                    ⏱ Estimated Wait: {estimatedWaitTime}
                </span>
            </div>

            {/* 3. Smart Alerts */}
            {smartAlerts && smartAlerts.length > 0 && (
                <div style={styles.alertBox}>
                    <strong>💡 Suggestions:</strong>
                    <ul style={{ margin: '5px 0 0 20px', padding: 0 }}>
                        {smartAlerts.map((alert: string, idx: number) => (
                            <li key={idx}>{alert}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};
